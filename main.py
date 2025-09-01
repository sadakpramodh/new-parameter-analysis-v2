# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import io, re
from typing import Tuple, List, Optional
import plotly.graph_objects as go

# =========================================
# Page
# =========================================
st.set_page_config(page_title="CCM Quality Bands", layout="wide")

# =========================================
# Tunables
# =========================================
MIN_SAMPLES_CLASS = 30
TARGET_COVERAGE   = 0.30
MAX_BINS          = 50
SMALL_SAMPLE_BAND = 0.50
EPS               = 1e-9

DEFECTS_WHITELIST = ['Cutmark', 'Thin Socket', 'Extractor Crack', 'Core Burst']

# Only allow these parameters (order preserved)
ALLOWED_PARAMS = [
    "Hopper_Speed2_Time","Core_Loader_In","Hopper_Speed1_Time","Casting_Time",
    "Matel_Flow_Time_Chute_To_Socket","Socket_Filling_Time","Hopper_Speed2",
    "Hopper_Speed3_Time","Hopper_Speed1","Hopper_Speed3","Core_Loader_Out",
    "R_Runner_Distance","Extractor_In_Time","L_Runner_Distance","Boggi_Speed1",
    "Extractor_JAW_Expansion_Position","Socket_Speed","Boggi_Speed1_Position",
    "Extractor_JAW_Close_Position","Outlet_Water_Temperature",
    "Extractor_Forward_Current","Extractor_Backward_Current",
    "Extractor_Back_Slow_Speed","Extractor_Back_High_Speed","Extractor_JAW_Open_Position",
    "Casting_Delay"
]

# =========================================
# Utilities
# =========================================
def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')

def tiered_subset(df: pd.DataFrame, dn, pipe_class) -> Tuple[pd.DataFrame, str]:
    s1 = df[(df['DN']==dn) & (df['Pipe_Class']==pipe_class)].copy()
    if len(s1) >= MIN_SAMPLES_CLASS: return s1, 'Class'
    s2 = df[df['DN']==dn].copy()
    if len(s2) >= MIN_SAMPLES_CLASS: return s2, 'DN'
    return df.copy(), 'Global'

def iqr_band(series: pd.Series, frac: float=0.50) -> Tuple[float,float]:
    lo = (1-frac)/2; hi = 1-lo
    ql, qh = series.quantile(lo), series.quantile(hi)
    return float(ql), float(qh)

def build_bins(filtered: pd.DataFrame, param: str, max_bins: int=MAX_BINS):
    n = len(filtered)
    q = max(5, min(max_bins, int(np.sqrt(max(n,1)))))
    try:
        b = pd.qcut(filtered[param], q=q, duplicates='drop')
        if not b.isna().all(): return b, 'qcut'
    except Exception:
        pass
    uniq = filtered[param].nunique(dropna=True)
    k = max(3, min(max_bins, uniq if uniq>=3 else 3))
    try:
        b = pd.cut(filtered[param], bins=k, include_lowest=True, duplicates='drop')
        if not b.isna().all(): return b, 'cut'
    except Exception:
        pass
    return pd.Series([pd.NA]*n, index=filtered.index), 'none'

def contiguous_regions_by_prob(triples: List[Tuple[float,float,float]], thr: float):
    regions, cur = [], []
    for left,right,p in triples:
        if p <= thr + EPS:
            cur = [left,right] if not cur else [cur[0], right]
        else:
            if cur: regions.append(tuple(cur)); cur=[]
    if cur: regions.append(tuple(cur))
    return regions

def lowest_rejection_cover_band(gb_idx: pd.DataFrame, filtered: pd.DataFrame, param: str, min_cov: float=TARGET_COVERAGE):
    bins = list(gb_idx.index.categories)
    if not bins: return None, None, 0.0
    totals  = np.array([gb_idx.loc[b,'total']    if b in gb_idx.index else 0 for b in bins], float)
    rejects = np.array([gb_idx.loc[b,'rejected'] if b in gb_idx.index else 0 for b in bins], float)
    N, totalN = len(bins), totals.sum()
    best = (None,None,1.0)
    for i in range(N):
        for j in range(i,N):
            cov  = totals[i:(j+1)].sum() / max(totalN,1.0)
            if cov >= min_cov:
                mprob = (rejects[i:(j+1)].sum() / np.clip(totals[i:(j+1)].sum(), 1, None))
                if (best[0] is None) or (mprob < best[2]): best=(i,j,mprob)
    if best[0] is None: return None,None,0.0
    LCL,UCL = float(bins[best[0]].left), float(bins[best[1]].right)
    coverage = totals[best[0]:(best[1]+1)].sum()/max(totalN,1.0)
    return LCL,UCL,float(coverage)

def compute_band_and_table(df_in: pd.DataFrame, param: str, dn, pipe_class, class_thr: float):
    summary = {"LCL":None,"UCL":None,"has_limits":False,"method":"","data_tier":"",
               "n_filtered":0,"coverage_pct":0.0,"avg_rej_inside":None,"notes":[]}

    subset, data_tier = tiered_subset(df_in, dn, pipe_class)
    summary["data_tier"] = data_tier
    if param not in subset.columns:
        summary["notes"].append(f"Parameter '{param}' not present"); return summary, pd.DataFrame()

    subset = subset.copy(); subset[param] = to_numeric(subset[param])
    s_nonan = subset[param].dropna()
    if s_nonan.empty:
        summary["notes"].append("All-NaN/non-numeric"); return summary, pd.DataFrame()

    Q1,Q3 = s_nonan.quantile(0.25), s_nonan.quantile(0.75); IQR = Q3-Q1
    filtered = subset[(subset[param] >= Q1-1.5*IQR) & (subset[param] <= Q3+1.5*IQR)].copy()
    if filtered.empty:
        filtered = subset.dropna(subset=[param]).copy()
        summary["notes"].append("IQR removed all; using unfiltered values")

    n = len(filtered); summary["n_filtered"] = int(n)

    if n < MIN_SAMPLES_CLASS:
        lo,hi = iqr_band(filtered[param], frac=SMALL_SAMPLE_BAND)
        summary.update({"LCL":lo,"UCL":hi,"has_limits":True,"method":"small-sample-IQR","coverage_pct":50.0})
        # provisional in-slice mean (we overwrite later with the global-denominator band %)
        seg = filtered[(filtered[param]>=lo) & (filtered[param]<=hi)]
        if not seg.empty: summary["avg_rej_inside"] = float(seg['Rejected_Flag'].mean()*100)
        return summary, pd.DataFrame()

    binned, method_used = build_bins(filtered, param)
    if method_used=='none' or binned.isna().all():
        lo,hi = iqr_band(filtered[param], frac=0.50)
        summary.update({"LCL":lo,"UCL":hi,"has_limits":True,"method":"IQR-fallback","coverage_pct":50.0})
        seg = filtered[(filtered[param]>=lo) & (filtered[param]<=hi)]
        if not seg.empty: summary["avg_rej_inside"] = float(seg['Rejected_Flag'].mean()*100)
        return summary, pd.DataFrame()

    filtered['bin'] = binned
    gb = (filtered.groupby('bin', observed=False)
                    .agg(total=('Rejected_Flag','count'), rejected=('Rejected_Flag','sum'))
                    ).reset_index()
    gb['rejection_prob'] = gb['rejected']/gb['total']
    gb_idx = gb.set_index('bin')

    edges   = gb_idx.index.categories
    triples = [(b.left,b.right,float(gb_idx.loc[b,'rejection_prob'])) for b in edges if b in gb_idx.index]

    regions = contiguous_regions_by_prob(triples, class_thr)
    if regions:
        LCL,UCL = max(regions, key=lambda r: r[1]-r[0])
        summary.update({"LCL":LCL,"UCL":UCL,"has_limits":True,"method":method_used})
        tot = gb['total'].sum()
        cov = gb[(gb['bin'].apply(lambda x: x.left>=LCL and x.right<=UCL))]['total'].sum()/max(tot,1.0)
        summary["coverage_pct"] = round(float(cov)*100,2)
    else:
        L2,U2,cov2 = lowest_rejection_cover_band(gb_idx, filtered, param, min_cov=TARGET_COVERAGE)
        if L2 is not None:
            summary.update({"LCL":L2,"UCL":U2,"has_limits":True,"method":f"{method_used}+low-reject-cover","coverage_pct":round(float(cov2)*100,2)})
            summary["notes"].append(f"No bin ≤ threshold; used lowest-reject cover ≥ {int(TARGET_COVERAGE*100)}%")
        else:
            lo,hi = iqr_band(filtered[param], frac=0.50)
            summary.update({"LCL":lo,"UCL":hi,"has_limits":True,"method":f"{method_used}+IQR-tertiary","coverage_pct":50.0})
            summary["notes"].append("No acceptable band; used IQR band")

    rows = []
    for b in edges:
        if b not in gb_idx.index: continue
        row = gb_idx.loc[b]; prob = float(row['rejection_prob'])
        in_band = (summary["has_limits"] and (b.left>=summary["LCL"] and b.right<=summary["UCL"]))
        label   = 'Green' if in_band else ('Red' if prob > class_thr + EPS else 'Amber')
        total = int(row['total']); rejected = int(row['rejected']); green = max(total-rejected,0)
        rows.append({
            'Parameter Range': f"{round(b.left,4)} – {round(b.right,4)}",
            'Pipes': total, 'Green Pipes': green, 'Rejected Pipes': rejected,
            'Rejection Probability (%)': round(prob*100,2),
            'Color': label, 'bin_left': float(b.left), 'bin_right': float(b.right)
        })
    just_df = pd.DataFrame(rows)

    # provisional in-slice mean (we overwrite in reporting with the global-denominator band %)
    if summary["has_limits"] and summary["LCL"] is not None and summary["UCL"] is not None:
        seg = filtered[(filtered[param]>=summary["LCL"]) & (filtered[param]<=summary["UCL"])]
        if not seg.empty: summary["avg_rej_inside"] = float(seg['Rejected_Flag'].mean()*100)

    return summary, just_df

def make_histogram_bins(just_df: pd.DataFrame):
    bins=[]
    for _,r in just_df.iterrows():
        color = '#34a853' if r['Color']=='Green' else ('#ea4335' if r['Color']=='Red' else '#fbbc05')
        bins.append({'x0':r['bin_left'],'x1':r['bin_right'],'color':color})
    return bins

# =========================================
# Sidebar – Upload
# =========================================
with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("CSV exported from CCM (must include DN, Pipe_Class, Rejected_Flag)", type=["csv"])
    st.markdown("**Required columns**: `DN`, `Pipe_Class`, `Rejected_Flag`")

if not up:
    st.title("CCM – Data-Driven Green/Red Bins for Process Parameters")
    st.info("Upload a CSV to proceed.")
    st.stop()

# CCM id from file name
fname   = up.name or ""
m = re.search(r'CCM\s*([0-9]{1,2})', fname, flags=re.IGNORECASE)
ccm_tag = f"CCM{m.group(1)}" if m else None

# Title with CCM tag
title_suffix = f" • {ccm_tag}" if ccm_tag else ""
st.title(f"CCM – Data-Driven Green/Red Bins for Process Parameters{title_suffix}")
if ccm_tag:
    st.caption(f"Showing results for **{ccm_tag}** (parsed from file name: {fname})")

# =========================================
# Load & normalize
# =========================================
df_raw = pd.read_csv(up, low_memory=False)
df_raw.columns = df_raw.columns.str.strip()

missing = [c for c in ['DN','Pipe_Class','Rejected_Flag'] if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_raw['Rejected_Flag'] = pd.to_numeric(df_raw['Rejected_Flag'], errors='coerce').fillna(0).clip(0,1)

df = df_raw.copy()
df['Pipe_Class'] = df['Pipe_Class'].astype(str).str.strip().replace({'C40':'K7+C40','K7':'K7+C40'})

# Visual defect presence + canonicalization
has_defect = 'VISUAL DEFECT' in df.columns
if has_defect:
    _canon_map = {
        'cutmark': 'Cutmark',
        'thin socket': 'Thin Socket',
        'extractor crack': 'Extractor Crack',
        'core bust': 'Core Burst',
        'core burst': 'Core Burst',
        'corebust': 'Core Burst',
    }
    raw = df['VISUAL DEFECT'].astype(str).str.strip().str.lower()
    df['VISUAL DEFECT'] = raw.map(_canon_map).fillna(df['VISUAL DEFECT'].astype(str).str.strip())

# =========================================
# Filters
# =========================================
all_dns     = sorted(df['DN'].dropna().unique().tolist())
all_classes = sorted(df['Pipe_Class'].dropna().unique().tolist())

c1,c2 = st.columns(2)
with c1: dn_sel = st.selectbox("Filter: DN", options=all_dns, index=0)
with c2: pc_sel = st.selectbox("Filter: Pipe Class (C40 & K7 merged as K7+C40)", options=all_classes, index=0)

# Parameters selection (restricted to ALLOWED_PARAMS present & numeric)
non_candidate = {'DN','Pipe_Class','Rejected_Flag','VISUAL DEFECT'}
allowed_in_df = [p for p in ALLOWED_PARAMS if p in df.columns]
candidates=[]
for c in allowed_in_df:
    if c in non_candidate: continue
    s = pd.to_numeric(df[c], errors='coerce')
    if s.notna().mean()>0.5: candidates.append(c)

if not candidates:
    st.error("None of the allowed parameters were found (or numeric).")
    st.stop()

params_selected = st.multiselect("Parameter(s) to analyze", options=candidates, default=candidates)
if not params_selected:
    st.warning("Select at least one parameter.")
    st.stop()

# We always compute All Defects + each individual defect (if present)
if has_defect:
    wanted = ['Cutmark','Core Burst','Extractor Crack','Thin Socket']
    available = [d for d in wanted if d in df['VISUAL DEFECT'].unique()]
    defect_slices: List[Optional[str]] = [None] + available  # None => All defects
    st.caption("Cumulative table computes **All Defects** separately plus each of: "
               "**Cutmark, Core Burst, Extractor Crack, Thin Socket**.")
else:
    defect_slices = [None]

# Summary pills
st.markdown("---")
m1,m2 = st.columns(2)
m1.metric("Selected DN", dn_sel)
m2.metric("Selected Pipe Class", pc_sel)

# =========================================
# Compute results
# =========================================
result_blocks = []
cumulative_rows = []

for defect_sel in defect_slices:
    if has_defect and (defect_sel is not None):
        work_df = df[df['VISUAL DEFECT']==defect_sel].copy()
        defect_label = defect_sel
    else:
        work_df = df.copy()
        defect_label = "All Defects" if has_defect else "—"

    # class threshold for this slice
    class_mask_slice = (work_df['DN']==dn_sel) & (work_df['Pipe_Class']==pc_sel)
    class_threshold = work_df.loc[class_mask_slice, 'Rejected_Flag'].mean() if class_mask_slice.any() else work_df['Rejected_Flag'].mean()

    # subset for graphs (DN+Class within this slice)
    plot_df_base = work_df[(work_df['DN']==dn_sel) & (work_df['Pipe_Class']==pc_sel)].copy()

    # ------- Rejection % (Defect) OUT OF ALL pipes in DN+Class -------
    dnclass_df = df[(df['DN'] == dn_sel) & (df['Pipe_Class'] == pc_sel)].copy()
    if dnclass_df.empty:
        rej_pct_defect = None
    else:
        if has_defect and (defect_sel is not None):
            # numerator: number of rejected pipes with THIS defect (DN+Class)
            num_rej_defect = dnclass_df.loc[dnclass_df['VISUAL DEFECT'] == defect_sel, 'Rejected_Flag'].sum()
            rej_pct_defect = (float(num_rej_defect) / float(len(dnclass_df))) * 100.0
        else:
            # All Defects = overall rejection rate in DN+Class
            rej_pct_defect = float(dnclass_df['Rejected_Flag'].mean() * 100.0)

    for param in params_selected:
        summary, just_df = compute_band_and_table(work_df, param, dn_sel, pc_sel, class_threshold)

        # ------- Rejection % (Band) against ALL pipes in DN+Class WITHIN [LCL,UCL] -------
        if summary["has_limits"] and (summary["LCL"] is not None) and (summary["UCL"] is not None):
            dnclass_param = dnclass_df.copy()
            dnclass_param[param] = pd.to_numeric(dnclass_param[param], errors='coerce')
            band_mask = dnclass_param[param].between(float(summary["LCL"]), float(summary["UCL"]), inclusive="both")

            denom = int(band_mask.sum())  # all pipes in DN+Class inside the band
            if denom > 0:
                if has_defect and (defect_sel is not None):
                    # numerator: rejected pipes WITH THIS DEFECT inside the band
                    numer = int(dnclass_param.loc[(dnclass_param['VISUAL DEFECT']==defect_sel) & band_mask, 'Rejected_Flag'].sum())
                else:
                    # All Defects: any rejection inside the band
                    numer = int(dnclass_param.loc[band_mask, 'Rejected_Flag'].sum())
                rej_pct_band = (numer / denom) * 100.0
            else:
                rej_pct_band = None
        else:
            rej_pct_band = None

        # ---- cumulative row (LCL/UCL + rejection% per defect + band) ----
        cumulative_rows.append({
            "CCM": ccm_tag or "",
            "DN": dn_sel,
            "Pipe_Class": pc_sel,
            "Visual_Defect": defect_label,
            "Parameter": param,
            "LCL": None if summary["LCL"] is None else float(summary["LCL"]),
            "UCL": None if summary["UCL"] is None else float(summary["UCL"]),
            "Rejection % (Defect)": None if rej_pct_defect is None else round(rej_pct_defect, 2),
            "Rejection % (Band)": None if rej_pct_band is None else round(float(rej_pct_band), 2)
        })

        # For the metrics card below, display the corrected band % (global denominator)
        summary["avg_rej_inside"] = rej_pct_band

        # ---- figure scaffold (bars + shading + LCL/UCL) ----
        vals_all   = pd.to_numeric(plot_df_base[param], errors='coerce')
        vals_green = vals_all[plot_df_base['Rejected_Flag']==0].dropna()
        vals_rej   = vals_all[plot_df_base['Rejected_Flag']==1].dropna()

        fig = go.Figure()
        avg_bin_w = None

        if not just_df.empty:
            centers = (just_df['bin_left'] + just_df['bin_right'])/2.0
            widths  = (just_df['bin_right'] - just_df['bin_left']).abs()
            counts_rej   = just_df['Rejected Pipes']
            counts_green = (just_df['Pipes'] - just_df['Rejected Pipes']).clip(lower=0)

            fig.add_bar(x=centers, y=counts_green, width=widths, name='Green (Accepted)', marker_color='green', opacity=0.5)
            fig.add_bar(x=centers, y=counts_rej,   width=widths, name='Rejected (Count)', marker_color='orange', opacity=0.5)

            for b in make_histogram_bins(just_df):
                fig.add_vrect(x0=b['x0'], x1=b['x1'], fillcolor=b['color'], opacity=0.20, line_width=0, layer='below')

            avg_bin_w = float(widths.mean()) if len(widths) else None
        else:
            if len(vals_green)>0: fig.add_histogram(x=vals_green, nbinsx=60, name='Green (Accepted)', opacity=0.45, marker_color='green')
            if len(vals_rej)>0:   fig.add_histogram(x=vals_rej,   nbinsx=60, name='Rejected (Count)', opacity=0.45, marker_color='orange')
            vmin = np.nanmin([vals_green.min() if len(vals_green)>0 else np.nan,
                              vals_rej.min()   if len(vals_rej)>0   else np.nan])
            vmax = np.nanmax([vals_green.max() if len(vals_green)>0 else np.nan,
                              vals_rej.max()   if len(vals_rej)>0   else np.nan])
            if np.isfinite(vmin) and np.isfinite(vmax) and float(vmax)>float(vmin):
                avg_bin_w = (float(vmax)-float(vmin))/60.0

        if summary['LCL'] is not None:
            fig.add_vline(x=summary['LCL'], line_width=2, line_dash='dash', line_color='green')
            fig.add_annotation(x=summary['LCL'], y=1.02, yref='paper',
                               text=f"LCL: {float(summary['LCL']):.2f}",
                               showarrow=False, font=dict(color='green', size=12),
                               bgcolor='rgba(0,128,0,0.10)', xanchor='left')
        if summary['UCL'] is not None:
            fig.add_vline(x=summary['UCL'], line_width=2, line_dash='dash', line_color='green')
            fig.add_annotation(x=summary['UCL'], y=1.02, yref='paper',
                               text=f"UCL: {float(summary['UCL']):.2f}",
                               showarrow=False, font=dict(color='green', size=12),
                               bgcolor='rgba(0,128,0,0.10)', xanchor='right')

        # package for bottom rendering
        result_blocks.append({
            "defect": defect_label, "param": param, "summary": summary,
            "fig": fig, "vals_green": vals_green, "vals_rej": vals_rej, "avg_bin_w": avg_bin_w
        })

# =========================================
# Cumulative table FIRST
# =========================================
st.markdown("---")
st.subheader("Cumulative Results (LCL, UCL, Rejection % by Defect and Band)")

cum_df = pd.DataFrame(cumulative_rows)
cum_cols = ["CCM","DN","Pipe_Class","Visual_Defect","Parameter","LCL","UCL","Rejection % (Defect)","Rejection % (Band)"]
cum_df = cum_df[cum_cols]
st.dataframe(cum_df, use_container_width=True)

# Download cumulative to Excel
out_xlsx = io.BytesIO()
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    cum_df.to_excel(writer, index=False, sheet_name="Cumulative_Limits")
st.download_button(
    "Download Cumulative Results (Excel)",
    data=out_xlsx.getvalue(),
    file_name=f"Cumulative_Limits_{ccm_tag or 'CCM'}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Info notes if any
notes = sorted({n for rb in result_blocks for n in (rb["summary"].get("notes") or [])})
if notes:
    st.info("; ".join(notes))

# =========================================
# Charts & Detailed Statistics (BOTTOM)
# =========================================
st.markdown("---")
st.subheader("Charts & Detailed Statistics")

# Global curve visibility
colg, colr = st.columns(2)
with colg:
    green_modes = st.multiselect("Green curves", ["Gaussian","KDE"], default=["Gaussian","KDE"], key="g_global")
with colr:
    reject_modes = st.multiselect("Rejected curves", ["Gaussian","KDE"], default=["Gaussian","KDE"], key="r_global")

def add_scaled_curves(fig, x_grid, yg_norm, yg_kde, yr_norm, yr_kde):
    if (yg_norm is not None) and ("Gaussian" in green_modes):
        fig.add_scatter(x=x_grid, y=yg_norm, mode='lines', name='Green – Gaussian',
                        line=dict(color='green', width=2, dash='solid'))
    if (yg_kde is not None) and ("KDE" in green_modes):
        fig.add_scatter(x=x_grid, y=yg_kde, mode='lines', name='Green – KDE',
                        line=dict(color='green', width=2, dash='dot'))
    if (yr_norm is not None) and ("Gaussian" in reject_modes):
        fig.add_scatter(x=x_grid, y=yr_norm, mode='lines', name='Rejected – Gaussian',
                        line=dict(color='orange', width=2, dash='solid'))
    if (yr_kde is not None) and ("KDE" in reject_modes):
        fig.add_scatter(x=x_grid, y=yr_kde, mode='lines', name='Rejected – KDE',
                        line=dict(color='orange', width=2, dash='dot'))

for rb in result_blocks:
    fig = rb["fig"]
    vals_green, vals_rej, avg_w = rb["vals_green"], rb["vals_rej"], rb["avg_bin_w"]

    # x-grid for curves
    x_vals = []
    if len(vals_green)>0: x_vals += [float(vals_green.min()), float(vals_green.max())]
    if len(vals_rej)>0:   x_vals += [float(vals_rej.min()),   float(vals_rej.max())]
    if not x_vals: x_vals=[0.0,1.0]
    x_min, x_max = min(x_vals), max(x_vals)
    if x_max==x_min: x_max = x_min + 1.0
    x_grid = np.linspace(x_min, x_max, 300)

    # scaled Gaussian
    def scaled_normal(values):
        if len(values)<2: return None
        mu=float(values.mean()); sigma=float(values.std(ddof=0))
        if sigma<=0 or not np.isfinite(sigma): return None
        N=len(values)
        bw = avg_w if (avg_w is not None and np.isfinite(avg_w) and avg_w>0) else (x_max-x_min)/60.0
        pdf = (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_grid-mu)/sigma)**2)
        return N*bw*pdf

    # scaled KDE
    def scaled_kde(values):
        n=len(values)
        if n<2: return None
        vals=values.to_numpy(dtype=float); std=float(values.std(ddof=0))
        if not np.isfinite(std) or std<=0: return None
        h=1.06*std*(n**(-1/5))
        if not np.isfinite(h) or h<=0: return None
        u=(x_grid[:,None]-vals[None,:])/h
        phi=np.exp(-0.5*u**2)/np.sqrt(2*np.pi)
        f=phi.mean(axis=1)/h
        bw = avg_w if (avg_w is not None and np.isfinite(avg_w) and avg_w>0) else (x_max-x_min)/60.0
        return n*bw*f

    yg_norm = scaled_normal(vals_green)
    yg_kde  = scaled_kde(vals_green)
    yr_norm = scaled_normal(vals_rej)
    yr_kde  = scaled_kde(vals_rej)

    add_scaled_curves(fig, x_grid, yg_norm, yg_kde, yr_norm, yr_kde)

    fig.update_layout(
        title=f"Histogram & Curves • DN {dn_sel} • Class {pc_sel} • Defect: {rb['defect']} • Param: {rb['param']}",
        xaxis_title=rb['param'], yaxis_title='Count', legend=dict(orientation='h'),
        barmode='overlay', margin=dict(l=10,r=10,t=40,b=10)
    )

    # Metrics row
    s = rb["summary"]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Defect", rb["defect"])
    k2.metric("Parameter", rb["param"])
    k3.metric("Method", s['method'] or '—')
    k4.metric("Coverage %", f"{s['coverage_pct']:.2f}")
    k5.metric("Rejection % (Band)", f"{s['avg_rej_inside']:.2f}" if s['avg_rej_inside'] is not None else '—')

    st.plotly_chart(fig, use_container_width=True)

st.caption("Top table shows LCL/UCL and rejection percentages per (Defect × Parameter). Charts & detailed stats are below. C40 & K7 are merged as K7+C40. ‘Rejection % (Band)’ uses all pipes inside the band as denominator (DN+Class).")
