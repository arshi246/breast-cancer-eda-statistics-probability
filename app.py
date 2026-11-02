"""Breast Cancer Patient Outcomes - single-page Streamlit dashboard."""
from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statistics_section import render_statistics_probability

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CSV_NAME = "Breast_Cancer.csv"
RAW_NUMERIC_COLUMNS = ["Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive", "Survival Months"]
CATEGORICAL_COLUMNS = [
    "Race", "Marital Status", "T Stage", "N Stage", "6th Stage", "Differentiate", "Grade",
    "A Stage", "Estrogen Status", "Progesterone Status", "Status",
]
REQUIRED_COLUMNS = RAW_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

STATUS_ORDER = ("Alive", "Dead")
STAGE_ORDER = ("IIA", "IIB", "IIIA", "IIIB", "IIIC")
GRADE_ORDER = ("Grade 1", "Grade 2", "Grade 3", "Grade 4 (Anaplastic)")
HORMONE_ORDER = ("Double Positive (ER+/PR+)", "Single Positive", "Double Negative (ER-/PR-)")
AGE_GROUP_ORDER = ("30-39 (Young)", "40-49 (Early Mid)", "50-59 (Late Mid)", "60-69 (Senior)")
TUMOR_ORDER = ("Small (<=20 mm)", "Medium (21-50 mm)", "Large (>50 mm)")
SURVIVAL_BRACKET_LABELS = tuple(f"Yr {i}" for i in range(1, 10))
GRADE_MAP = {"1": "Grade 1", "2": "Grade 2", "3": "Grade 3", "anaplastic; Grade IV": "Grade 4 (Anaplastic)"}
SMALL_GROUP = 30

ORDERS = {
    "6th Stage": STAGE_ORDER, "Grade": GRADE_ORDER, "Hormone_Profile": HORMONE_ORDER,
    "Age_Group": AGE_GROUP_ORDER, "Tumor_Category": TUMOR_ORDER,
    "T Stage": ("T1", "T2", "T3", "T4"), "N Stage": ("N1", "N2", "N3"), "A Stage": ("Regional", "Distant"),
    "Differentiate": ("Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"),
    "Estrogen Status": ("Positive", "Negative"), "Progesterone Status": ("Positive", "Negative"),
}
LABELS = {
    "6th Stage": "6th stage", "Grade": "grade", "Hormone_Profile": "hormone profile", "N Stage": "N stage",
    "T Stage": "T stage", "Age_Group": "age group", "Tumor_Category": "tumor size category", "Race": "race",
    "Marital Status": "marital status", "Differentiate": "differentiation", "A Stage": "A stage",
    "Estrogen Status": "estrogen status", "Progesterone Status": "progesterone status",
}
DIST_TITLES = {
    "Age": "Age at diagnosis (years)", "Tumor Size": "Tumor size (mm)",
    "Regional Node Examined": "Regional nodes examined", "Regional Node Positive": "Regional nodes positive",
    "Lymph_Node_Ratio": "Lymph-node ratio", "Survival Months": "Survival months",
}
CORR_COLUMNS = {
    "Age": "Age", "Tumor Size": "Tumor size", "Regional Node Examined": "Nodes examined",
    "Regional Node Positive": "Nodes positive", "Survival Months": "Survival months",
    "Lymph_Node_Ratio": "Node ratio", "Status_Binary": "Dead (1/0)",
}
NOTEBOOK_BASELINE = [
    ("Raw records", 4024, "raw_rows"), ("Missing values", 0, "missing_values"),
    ("Alive in raw file", 3408, "raw_alive"), ("Dead in raw file", 616, "raw_dead"),
    ("Duplicates removed", 1, "duplicates_removed"), ("Rows after de-duplication", 4023, "rows_after_dedup"),
    ("IQR outliers removed", 580, "outliers_removed"), ("Clean rows", 3443, "clean_rows"),
    ("Alive after cleaning", 3021, "clean_alive"), ("Dead after cleaning", 422, "clean_dead"),
]

COLORS = {
    "teal": "#16877F", "red": "#E4686B", "blue": "#3D6280", "amber": "#E3A04F",
    "ink": "#17324D", "muted": "#64748B", "grid": "#EDF1F6", "line": "#DCE3EC",
}
STATUS_COLORS = {"Alive": COLORS["teal"], "Dead": COLORS["red"]}
FONT = "Source Sans 3, Source Sans Pro, Segoe UI, sans-serif"
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


# --------------------------------------------------------------------------- #
# Data pipeline
# --------------------------------------------------------------------------- #
def prepare_dataset(raw_df: pd.DataFrame):
    """Notebook-aligned cleaning. Returns (cleaned_df, outlier_audit_df, metadata)."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Reginol Node Positive": "Regional Node Positive", "differentiate": "Differentiate"})
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    raw_rows = len(df)

    for col in RAW_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing_values = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    if df.empty:
        raise ValueError("No usable rows after removing missing values.")
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    bad_status = set(df["Status"].unique()) - set(STATUS_ORDER)
    if bad_status:
        raise ValueError(f"Unexpected Status values: {sorted(bad_status)}")
    for col in ("Estrogen Status", "Progesterone Status"):
        bad = set(df[col].unique()) - {"Positive", "Negative"}
        if bad:
            raise ValueError(f"Unexpected {col} values: {sorted(bad)}")

    df["Grade"] = df["Grade"].replace(GRADE_MAP)
    raw_alive = int((df["Status"] == "Alive").sum())
    raw_dead = int((df["Status"] == "Dead").sum())

    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(df)

    audit_rows = []
    keep_all = pd.Series(True, index=df.index)
    for col in RAW_NUMERIC_COLUMNS:
        q1, q3 = df[col].quantile([0.25, 0.75])   # hamesha poore (de-duplicated) data par
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep = df[col].between(lower, upper)
        audit_rows.append({
            "Column": col, "Q1": q1, "Q3": q3, "Lower bound": lower, "Upper bound": upper,
            "Rows removed": int((~keep).sum()),
        })
        keep_all &= keep
    cleaned = df[keep_all].reset_index(drop=True)


    exam = cleaned["Regional Node Examined"]
    cleaned["Lymph_Node_Ratio"] = (cleaned["Regional Node Positive"] / exam.replace(0, np.nan)).fillna(0.0)
    cleaned["Survival_Years"] = cleaned["Survival Months"] / 12
    cleaned["Age_Group"] = pd.cut(cleaned["Age"], bins=[29, 39, 49, 59, 70], labels=list(AGE_GROUP_ORDER)).astype(object)
    cleaned["Tumor_Category"] = pd.cut(cleaned["Tumor Size"], bins=[0, 20, 50, 100], labels=list(TUMOR_ORDER)).astype(object)
    er = cleaned["Estrogen Status"].eq("Positive")
    pr = cleaned["Progesterone Status"].eq("Positive")
    cleaned["Hormone_Profile"] = np.select(
        [er & pr, ~er & ~pr], [HORMONE_ORDER[0], HORMONE_ORDER[2]], default=HORMONE_ORDER[1]
    )
    years = np.clip(np.ceil(cleaned["Survival Months"] / 12), 1, 9).astype(int)
    cleaned["Survival_Year_Bracket"] = "Yr " + years.astype(str)
    cleaned["Status_Binary"] = (cleaned["Status"] == "Dead").astype(int)

    meta = {
        "raw_rows": raw_rows, "missing_values": missing_values, "raw_alive": raw_alive, "raw_dead": raw_dead,
        "duplicates_removed": before_dedup - rows_after_dedup, "rows_after_dedup": rows_after_dedup,
        "outliers_removed": rows_after_dedup - len(cleaned), "clean_rows": len(cleaned),
        "clean_alive": int((cleaned["Status"] == "Alive").sum()), "clean_dead": int((cleaned["Status"] == "Dead").sum()),
    }
    return cleaned, pd.DataFrame(audit_rows), meta


def apply_filters(data: pd.DataFrame, races, stages, age_groups) -> pd.DataFrame:
    if not len(races) or not len(stages) or not len(age_groups):
        return data.iloc[0:0].copy()
    mask = data["Race"].isin(races) & data["6th Stage"].isin(stages) & data["Age_Group"].isin(age_groups)
    return data[mask].copy()


def compute_mortality_summary(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = data.groupby(group_col, observed=True)["Status_Binary"].agg(Patients="size", Deaths="sum").reset_index()
    g["Mortality"] = g["Deaths"] / g["Patients"] * 100
    order = list(ORDERS.get(group_col, ()))
    if order:
        order += [v for v in g[group_col].unique() if v not in order]
        g[group_col] = pd.Categorical(g[group_col], categories=order, ordered=True)
        g = g.sort_values(group_col)
        g[group_col] = g[group_col].astype(str)
    else:
        g = g.sort_values(group_col)
    return g.reset_index(drop=True)


def observed_survival_curve(data: pd.DataFrame, horizon: int = 60, reference_max_month: int = 107):
    """Crude observed share without a recorded death by month t (not Kaplan-Meier)."""
    horizon = int(min(horizon, reference_max_month))
    months = np.arange(0, horizon + 1)
    n = len(data)
    nan = float("nan")
    if n == 0:
        empty = pd.DataFrame({"Month": months, "Survival (%)": nan, "Cumulative deaths": 0})
        return empty, {
            "patients": 0, "deaths": 0, "horizon": horizon, "survival_at_horizon": nan,
            "deaths_by_horizon": 0, "deaths_by_horizon_share": nan, "beyond_patients": 0,
            "mortality_after_horizon": nan,
        }
    dead_months = np.sort(data.loc[data["Status_Binary"] == 1, "Survival Months"].to_numpy(dtype=float))
    cum = np.searchsorted(dead_months, months, side="right")
    curve = pd.DataFrame({"Month": months, "Survival (%)": 100 * (1 - cum / n), "Cumulative deaths": cum})
    deaths = len(dead_months)
    by_h = int(cum[-1])
    beyond = data[data["Survival Months"] > horizon]
    metrics = {
        "patients": n, "deaths": deaths, "horizon": horizon,
        "survival_at_horizon": float(curve["Survival (%)"].iloc[-1]),
        "deaths_by_horizon": by_h,
        "deaths_by_horizon_share": (by_h / deaths * 100) if deaths else nan,
        "beyond_patients": len(beyond),
        "mortality_after_horizon": float(beyond["Status_Binary"].mean() * 100) if len(beyond) else nan,
    }
    return curve, metrics


def point_biserial_summary(data: pd.DataFrame, columns=None) -> pd.DataFrame:
    columns = list(columns or CORR_COLUMNS.keys())
    columns = [c for c in columns if c != "Status_Binary"]
    rows = []
    for col in columns:
        r = p = float("nan")
        if data["Status_Binary"].nunique() > 1 and data[col].nunique() > 1:
            from scipy.stats import pointbiserialr

            res = pointbiserialr(data["Status_Binary"], data[col])
            r, p = float(res[0]), float(res[1])
        a = abs(r)
        strength = "n/a" if np.isnan(r) else "Negligible" if a < 0.1 else "Weak" if a < 0.3 else "Moderate" if a < 0.5 else "Strong"
        direction = "n/a" if np.isnan(r) else ("Higher in Dead" if r > 0 else "Higher in Alive")
        rows.append({"Variable": col, "r": r, "p_value": p, "Strength": strength, "Direction": direction})
    out = pd.DataFrame(rows)
    out["_abs"] = out["r"].abs()
    return out.sort_values("_abs", ascending=False, na_position="last").drop(columns="_abs").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Insights (finding + recommendation)
# --------------------------------------------------------------------------- #
GRADIENT_ACTIONS = {
    "Stage gradient": "Prioritise {hi} for closer review and follow-up planning",
    "Grade gradient": "Treat {hi} as the high-risk grade and flag these cases early",
    "Hormone profile": "Check receptor status early; {hi} patients may need closer monitoring and clinician discussion",
    "Nodal stage": "Track node involvement closely; {hi} has the highest mortality, so review nodal status alongside stage",
}


def _gradient(summary, col, label, tone="red", min_n=10):
    s = summary[summary["Patients"] >= min_n]
    if len(s) < 2:
        return None
    lo, hi = s.loc[s["Mortality"].idxmin()], s.loc[s["Mortality"].idxmax()]
    action = GRADIENT_ACTIONS[label].format(hi=hi[col])
    if lo["Mortality"] > 0:
        action += f" (mortality is {hi['Mortality'] / lo['Mortality']:.1f}x the lowest group)"
    action += "."
    if hi["Patients"] < SMALL_GROUP:
        action += f" Note: only {int(hi['Patients'])} patients, so confirm before acting."
    return {
        "label": label, "tone": tone,
        "stat": f"{lo['Mortality']:.1f}% \u2192 {hi['Mortality']:.1f}%",
        "text": f"Lowest: {lo[col]} (n={int(lo['Patients']):,}). Highest: {hi[col]} (n={int(hi['Patients']):,}).",
        "action": action,
    }


AGE_ACTIONS = {
    "U-shaped": "Analyse the youngest and oldest groups separately; risk does not rise steadily with age.",
    "Rising with age": "Age looks like a steady risk factor; include it in any risk model.",
    "Falling with age": "Younger patients show higher mortality; investigate what differs in this group.",
    "Mixed": "No clean age trend; do not use age alone to judge risk.",
    "n/a": "Widen the filters to compare more age groups.",
}


def build_insights(data, summaries, metrics):
    n, deaths = len(data), int(data["Status_Binary"].sum())
    rate = deaths / n * 100
    base_action = (f"About 1 in {round(100 / rate)} patients has a recorded death. Use this as the baseline; "
                   "groups above it deserve attention.") if rate > 0 else "No recorded deaths in this view."
    items = [{
        "label": "Overall outcome", "tone": "teal", "stat": f"{rate:.1f}%",
        "text": f"{deaths:,} recorded deaths among {n:,} patients in view.", "action": base_action,
    }]
    for col, label in (("6th Stage", "Stage gradient"), ("Grade", "Grade gradient"),
                       ("Hormone_Profile", "Hormone profile"), ("N Stage", "Nodal stage")):
        item = _gradient(summaries[col], col, label)
        if item:
            items.append(item)

    age = summaries["Age_Group"]
    age = age[age["Patients"] >= 10]
    if len(age) >= 3:
        rates = age["Mortality"].to_list()
        mid_min = min(rates[1:-1])
        if rates[0] > mid_min and rates[-1] > mid_min:
            shape = "U-shaped"
        elif all(b >= a for a, b in zip(rates, rates[1:])):
            shape = "Rising with age"
        elif all(b <= a for a, b in zip(rates, rates[1:])):
            shape = "Falling with age"
        else:
            shape = "Mixed"
        text = ", ".join(f"{g.split(' ')[0]}: {r:.1f}%" for g, r in zip(age["Age_Group"], rates))
    else:
        shape, text = "n/a", "Not enough age groups in view."
    items.append({"label": "Age pattern", "tone": "blue", "stat": shape, "text": text, "action": AGE_ACTIONS[shape]})

    med = data.groupby("Status")[["Tumor Size", "Regional Node Positive", "Lymph_Node_Ratio"]].median()
    if {"Dead", "Alive"} <= set(med.index):
        d, a = med.loc["Dead"], med.loc["Alive"]
        higher = d["Tumor Size"] > a["Tumor Size"] or d["Regional Node Positive"] > a["Regional Node Positive"]
        action = ("Larger tumors and more positive nodes go with recorded deaths; use size and node count as early "
                  "risk flags in further analysis.") if higher else (
                  "Tumor size and node counts do not separate outcomes clearly here; look at stage and grade instead.")
        items.append({
            "label": "Median tumor size", "tone": "blue",
            "stat": f"{d['Tumor Size']:.0f} vs {a['Tumor Size']:.0f} mm",
            "text": f"Dead vs Alive. Positive nodes: {d['Regional Node Positive']:.0f} vs {a['Regional Node Positive']:.0f}; "
                    f"lymph-node ratio: {d['Lymph_Node_Ratio']:.2f} vs {a['Lymph_Node_Ratio']:.2f}.",
            "action": action,
        })
    if not np.isnan(metrics["survival_at_horizon"]):
        h = metrics["horizon"]
        after = metrics["mortality_after_horizon"]
        after_txt = f"; mortality after {h} months is {after:.2f}%" if not np.isnan(after) else ""
        share = metrics["deaths_by_horizon_share"]
        share_txt = f"{share:.1f}% of recorded deaths occur by {h} months" if not np.isnan(share) else "No recorded deaths"
        action = (f"Most deaths happen within {h} months, so concentrate follow-up effort in the first {h // 12} years; "
                  "risk is lower afterwards.") if (not np.isnan(share) and share >= 50) else (
                  "Deaths are spread across follow-up; keep long-term monitoring in place.")
        items.append({
            "label": f"{h}-month milestone", "tone": "teal", "stat": f"{metrics['survival_at_horizon']:.1f}%",
            "text": f"Crude observed survival at {h} months. {share_txt}{after_txt}.", "action": action,
        })
    return items


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _wrap_label(text, width=14):
    """Two-line axis label: 'Double Positive<br>(ER+/PR+)'."""
    text = str(text).replace("<=", "\u2264")
    if " (" in text:
        head, tail = text.split(" (", 1)
        return f"{head}<br>({tail}"
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _style(fig, height=340, legend=False):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT, size=13, color=COLORS["ink"]),
        margin=dict(l=4, r=12, t=56 if legend else 24, b=4), showlegend=legend, bargap=0.32,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, title_text="", font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=COLORS["line"]),
    )
    fig.update_xaxes(showgrid=False, linecolor=COLORS["line"], tickcolor=COLORS["line"], automargin=True,
                     title_standoff=10, tickfont=dict(size=12))
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, linecolor="rgba(0,0,0,0)", automargin=True,
                     title_standoff=10, tickfont=dict(size=12))
    return fig


def status_donut_figure(data):
    counts = data["Status"].value_counts().reindex(list(STATUS_ORDER)).fillna(0)
    fig = go.Figure(go.Pie(
        labels=list(counts.index), values=counts.to_numpy(), hole=0.64, sort=False,
        marker=dict(colors=[STATUS_COLORS[s] for s in counts.index], line=dict(color="white", width=3)),
        texttemplate="%{percent:.1%}", textfont=dict(color="white", size=13),
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(data):,}</b><br>patients", showarrow=False, font=dict(size=20, color=COLORS["ink"]))
    return _style(fig, 340, legend=True)


def count_plot_figure(data, column, height=340):
    counts = data[column].value_counts()
    order = list(ORDERS.get(column, ()))
    cats = [c for c in order if c in counts.index] + [c for c in counts.index if c not in order]
    vals = [int(counts[c]) for c in cats]
    total = sum(vals)
    fig = go.Figure(go.Bar(
        x=[_wrap_label(c) for c in cats], y=vals, marker_color=COLORS["teal"],
        text=[f"<b>{v:,}</b><br>{v / total * 100:.1f}%" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color=COLORS["ink"]), hovertemplate="%{customdata}<extra></extra>",
        customdata=[f"{c}: {v:,} patients ({v / total * 100:.1f}%)" for c, v in zip(cats, vals)],
    ))
    fig.update_xaxes(type="category")
    fig.update_yaxes(visible=False, range=[0, max(vals) * 1.3])
    return _style(fig, height)


def _kde(values, points=200):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) == 0:
        return None
    sd = v.std(ddof=1)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = 0.9 * spread * v.size ** -0.2
    if bw <= 0:
        return None
    xs = np.linspace(v.min(), v.max(), points)
    dens = np.exp(-0.5 * ((xs[:, None] - v[None, :]) / bw) ** 2).sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi))
    return xs, dens


def distribution_figure(data, column, height=340):
    v = data[column].to_numpy(dtype=float)
    integer_like = np.allclose(v, np.round(v)) and np.ptp(v) <= 45
    bins = np.arange(v.min() - 0.5, v.max() + 1.5) if integer_like else 30
    counts, edges = np.histogram(v, bins=bins)
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    fmt = ".2f" if column == "Lymph_Node_Ratio" else ".0f"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    fig = go.Figure(go.Bar(
        x=centers, y=counts, width=width * 0.94, marker_color=_rgba(COLORS["teal"], 0.78), name="Patients",
        hovertemplate="%{x:.2f}: %{y:,} patients<extra></extra>",
    ))
    fig.add_shape(type="line", x0=med, x1=med, y0=0, y1=1, yref="paper",
                  line=dict(color=COLORS["amber"], width=2, dash="dash"))
    right_side = med > (v.min() + v.max()) / 2
    fig.add_annotation(x=med, y=1, yref="paper", text=f"<b>Median {med:{fmt}}</b>", showarrow=False,
                       xanchor="right" if right_side else "left", yanchor="bottom",
                       xshift=-6 if right_side else 6, font=dict(size=12, color="#B9772B"))
    fig.update_xaxes(title_text=DIST_TITLES.get(column, column))
    fig.update_yaxes(title_text="Patients")
    fig = _style(fig, height, legend=False)
    fig.update_layout(margin=dict(l=4, r=12, t=40, b=4))
    chip = f"Median {med:{fmt}} \u00b7 IQR {q1:{fmt}}\u2013{q3:{fmt}}"
    return fig, chip


def mortality_rate_figure(summary, group_col, cohort_rate, height=360):
    cats = summary[group_col].astype(str).to_list()
    rates = summary["Mortality"].to_list()
    ns = summary["Patients"].to_list()
    colors = [COLORS["red"] if r > cohort_rate else COLORS["teal"] for r in rates]
    labels = [f"{_wrap_label(c)}<br><i>n={n:,}</i>" for c, n in zip(cats, ns)]
    text = [f"<b>{r:.1f}%</b>{'*' if n < SMALL_GROUP else ''}" for r, n in zip(rates, ns)]
    custom = list(zip(ns, summary["Deaths"], cats))
    fig = go.Figure(go.Bar(
        x=labels, y=rates, marker_color=colors, text=text, textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color=COLORS["ink"]), customdata=custom,
        hovertemplate="<b>%{customdata[2]}</b><br>Deaths: %{customdata[1]:,} of %{customdata[0]:,}<extra></extra>",
    ))
    fig.add_shape(type="line", y0=cohort_rate, y1=cohort_rate, x0=0, x1=1, xref="paper",
                  line=dict(color=COLORS["muted"], width=1.8, dash="dot"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0, max(max(rates), cohort_rate) * 1.25], ticksuffix="%")
    return _style(fig, height, legend=False)


def violin_box_figure(data, column, height=340):
    fig = go.Figure()
    for status in STATUS_ORDER:
        v = data.loc[data["Status"] == status, column]
        if v.empty:
            continue
        fig.add_trace(go.Violin(
            x=[status] * len(v), y=v, name=status, box_visible=True, meanline_visible=True, points=False,
            spanmode="hard", line_color=STATUS_COLORS[status], fillcolor=_rgba(STATUS_COLORS[status], 0.28), width=0.8,
        ))
    fig.update_yaxes(title_text=DIST_TITLES.get(column, column))
    return _style(fig, height)


def stage_tumor_box_figure(data, height=380):
    fig = go.Figure()
    for status in STATUS_ORDER:
        sub = data[data["Status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Box(x=sub["6th Stage"], y=sub["Tumor Size"], name=status, boxpoints=False,
                             marker_color=STATUS_COLORS[status], line=dict(width=1.6)))
    fig.update_layout(boxmode="group")
    fig.update_xaxes(categoryorder="array", categoryarray=[s for s in STAGE_ORDER if s in set(data["6th Stage"])],
                     title_text="6th stage")
    fig.update_yaxes(title_text="Tumor size (mm)")
    return _style(fig, height, legend=True)


def correlation_heatmap_figure(data, height=420):
    cols = list(CORR_COLUMNS)
    z = data[cols].corr().to_numpy()
    names = [CORR_COLUMNS[c] for c in cols]
    text = [["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=names, zmin=-1, zmax=1, text=text, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, "#5B8DB8"], [0.5, "#FFFFFF"], [1, COLORS["red"]]], xgap=2, ygap=2,
        colorbar=dict(thickness=12, len=0.85, outlinewidth=0, tickfont=dict(size=11)),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-35)
    return _style(fig, height)


def yearly_followup_figure(data, height=360):
    ct = pd.crosstab(data["Survival_Year_Bracket"], data["Status"]).reindex(
        index=list(SURVIVAL_BRACKET_LABELS), columns=list(STATUS_ORDER), fill_value=0)
    totals = ct.sum(axis=1)
    threshold = max(totals.max() * 0.06, 1)
    fig = go.Figure()
    for status in STATUS_ORDER:
        vals = ct[status]
        fig.add_trace(go.Bar(
            x=list(SURVIVAL_BRACKET_LABELS), y=vals, name=status, marker_color=STATUS_COLORS[status],
            text=[f"{v:,}" if v >= threshold else "" for v in vals], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=11), hovertemplate=f"{status}: %{{y:,}}<extra>%{{x}}</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=list(SURVIVAL_BRACKET_LABELS), y=totals, mode="text", text=[f"{v:,}" for v in totals],
        textposition="top center", textfont=dict(size=11, color=COLORS["muted"]), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(barmode="stack")
    fig.add_shape(type="line", x0=4.5, x1=4.5, y0=0, y1=1, yref="paper",
                  line=dict(color=COLORS["amber"], width=2, dash="dash"))
    fig.add_annotation(x=4.5, y=1, yref="paper", text="60-month milestone", showarrow=False, xanchor="left",
                       yanchor="bottom", xshift=6, font=dict(size=11, color="#B9772B"))
    fig.update_xaxes(categoryorder="array", categoryarray=list(SURVIVAL_BRACKET_LABELS))
    fig.update_yaxes(title_text="Patients", range=[0, max(totals.max(), 1) * 1.16])
    return _style(fig, height, legend=True)


def cumulative_survival_figure(curve, metrics, height=360):
    y = curve["Survival (%)"]
    lo = max(0.0, 5 * np.floor((y.min() - 2) / 5))
    last_x, last_y = int(curve["Month"].iloc[-1]), float(y.iloc[-1])
    fig = go.Figure(go.Scatter(
        x=curve["Month"], y=y, mode="lines", line=dict(color=COLORS["teal"], width=3, shape="hv"),
        fill="tozeroy", fillcolor=_rgba(COLORS["teal"], 0.1),
        hovertemplate="Month %{x}: %{y:.1f}% without recorded death<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[last_x], y=[last_y], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=10, color=COLORS["teal"], line=dict(color="white", width=2))))
    fig.add_annotation(x=last_x, y=last_y, text=f"<b>{last_y:.1f}%</b> at {last_x} months", showarrow=False,
                       xanchor="right", yanchor="top", xshift=-6, yshift=-14, font=dict(size=12, color=COLORS["ink"]))
    fig.update_xaxes(title_text="Months since diagnosis", tickvals=list(range(0, last_x + 1, 12)), range=[0, last_x])
    fig.update_yaxes(title_text="Observed survival", range=[lo, 100.5], ticksuffix="%")
    return _style(fig, height), lo


# --------------------------------------------------------------------------- #
# Cached loading and view building
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data(path_str: str, modified_time: float):
    try:
        raw = pd.read_csv(path_str, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(path_str, dtype=str, encoding="latin1")
    return prepare_dataset(raw)


@st.cache_data(show_spinner=False)
def export_bytes(path_str: str, modified_time: float):
    clean, audit, _ = load_data(path_str, modified_time)
    return clean.to_csv(index=False).encode("utf-8"), audit.to_csv(index=False).encode("utf-8")


MORT_NOTE = ("Red bars are above the overall average (dotted line), teal at or below. "
             "n = patients in the group.")
SMALL_NOTE = " * = fewer than 30 patients, so the rate is unstable."
DIST_NOTE = "Bars = patients per range. Dashed line = median (the middle value)."
COUNT_NOTE = "Number and share (%) of patients in each group."


def _card(title, fig, chip=None, caption=None):
    return {"title": title, "fig": fig, "chip": chip, "caption": caption}


@st.cache_data(show_spinner="Preparing charts...", max_entries=24)
def build_view(path_str: str, modified_time: float, races: tuple, stages: tuple, age_groups: tuple):
    clean, _, _ = load_data(path_str, modified_time)
    data = apply_filters(clean, races, stages, age_groups)
    view = {"n": len(data), "total": len(clean)}
    if data.empty:
        return view

    n, deaths = len(data), int(data["Status_Binary"].sum())
    rate = deaths / n * 100
    curve, metrics = observed_survival_curve(data)
    cards = {"status": _card("Recorded status", status_donut_figure(data), caption="Share of patients by recorded outcome.")}

    summaries = {}
    for col in ("6th Stage", "Grade", "Hormone_Profile", "N Stage", "T Stage", "Age_Group",
                "Tumor_Category", "Race", "Marital Status", "Differentiate", "A Stage"):
        summaries[col] = compute_mortality_summary(data, col)
        small = bool((summaries[col]["Patients"] < SMALL_GROUP).any())
        cards[f"mort:{col}"] = _card(
            f"Mortality by {LABELS[col]}", mortality_rate_figure(summaries[col], col, rate),
            chip=f"Overall {rate:.1f}%", caption=MORT_NOTE + (SMALL_NOTE if small else ""))

    for col in ("Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive", "Lymph_Node_Ratio"):
        fig, chip = distribution_figure(data, col)
        cards[f"dist:{col}"] = _card(DIST_TITLES[col], fig, chip=chip, caption=DIST_NOTE)
    for col in ("Age_Group", "Race", "Marital Status", "Tumor_Category", "T Stage", "N Stage", "6th Stage",
                "A Stage", "Grade", "Differentiate", "Estrogen Status", "Progesterone Status", "Hormone_Profile"):
        cards[f"count:{col}"] = _card(f"Patients by {LABELS[col]}", count_plot_figure(data, col), caption=COUNT_NOTE)

    for col in ("Tumor Size", "Regional Node Positive", "Lymph_Node_Ratio", "Age"):
        cards[f"violin:{col}"] = _card(f"{DIST_TITLES[col]} by status", violin_box_figure(data, col),
                                       caption="Violin = distribution shape; inner box = median and quartiles.")
    cards["box:stage-tumor"] = _card(
        "Tumor size across 6th stage, by status", stage_tumor_box_figure(data),
        caption="Box = middle 50% of patients; line inside = median. A higher box means larger tumors.")
    cards["heatmap"] = _card("Correlation heatmap", correlation_heatmap_figure(data), chip="Pearson r",
                             caption="Red = positive, blue = negative association. Closer to 0 = no link.")
    cards["yearly"] = _card("Yearly follow-up", yearly_followup_figure(data),
                            caption="Patients by recorded follow-up year. Totals sit above each bar.")
    surv_fig, lo = cumulative_survival_figure(curve, metrics)
    cards["survival"] = _card(
        "Cumulative observed survival", surv_fig, chip=f"{metrics['survival_at_horizon']:.1f}% at {metrics['horizon']} mo",
        caption=f"Y-axis zoomed to {lo:.0f}-100% for readability. Crude proportion, not Kaplan-Meier.")

    pb = point_biserial_summary(data)
    pb_show = pd.DataFrame({
        "Variable": [DIST_TITLES.get(v, v) for v in pb["Variable"]],
        "r": pb["r"].round(3),
        "p-value": ["n/a" if np.isnan(p) else "<0.001" if p < 0.001 else f"{p:.3f}" for p in pb["p_value"]],
        "Strength": pb["Strength"], "Direction": pb["Direction"],
    })

    view.update({
        "cards": cards, "metrics": metrics, "pb": pb_show, "insights": build_insights(data, summaries, metrics),
        "kpis": {"n": n, "deaths": deaths, "rate": rate, "median_fu": float(data["Survival Months"].median())},
        "view_csv": data.to_csv(index=False).encode("utf-8"),
    })
    return view


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root{--ink:#17324D;--muted:#64748B;--line:#DCE3EC;--bg:#F1F5F9;--teal:#16877F;--red:#E4686B;--blue:#3D6280;--amber:#E3A04F;
--shadow:0 1px 2px rgba(23,50,77,.05),0 8px 24px rgba(23,50,77,.06)}
html,body,[data-testid="stApp"],[data-testid="stMain"],section.main{scroll-behavior:smooth}
[data-testid="stHeader"]{background:var(--bg)!important}
#MainMenu,footer{visibility:hidden}
[data-testid="stMainBlockContainer"],.block-container{max-width:1240px;padding:4.2rem 2rem 4rem}

.hero{position:relative;overflow:hidden;color:#fff;border-radius:18px;padding:1.5rem 1.8rem;scroll-margin-top:5rem;
background:radial-gradient(circle at 92% 0%,rgba(255,255,255,.13) 0,rgba(255,255,255,0) 40%),linear-gradient(135deg,#0B2740 0%,#0F5A66 60%,#16877F 100%);
box-shadow:0 12px 28px rgba(11,39,64,.2)}
.hero-grid{display:grid;grid-template-columns:1.3fr 1fr;gap:1.6rem;align-items:center}
.hero-grid.solo{grid-template-columns:1fr}
.hero-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.26);
border-radius:999px;padding:.25rem .75rem;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.hero-badge i{width:7px;height:7px;border-radius:50%;background:#7EE6D8;box-shadow:0 0 0 3px rgba(126,230,216,.25)}
.hero-title{font-size:2.1rem;font-weight:800;line-height:1.12;letter-spacing:-.02em;margin:.7rem 0 .5rem}
.hero-title em{font-style:normal;color:#8CEBDD}
.hero-sub{max-width:560px;font-size:.98rem;line-height:1.55;opacity:.92;margin:0 0 .85rem}
.hero-tech span{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.03em;padding:.2rem .6rem;margin:0 .35rem .3rem 0;
border-radius:6px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2)}
.hero-stats{display:grid;grid-template-columns:repeat(2,1fr);gap:.6rem}
.hero-stat{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);border-radius:12px;padding:.65rem .9rem}
.hero-stat b{display:block;font-size:1.4rem;font-weight:800;letter-spacing:-.02em;line-height:1.1}
.hero-stat span{font-size:.74rem;opacity:.85;font-weight:600}

div[data-testid="stElementContainer"]:has(.nav-shell),div.element-container:has(.nav-shell){position:sticky;top:3.75rem;z-index:990;
background:var(--bg);padding:.45rem 0 .65rem}
.nav-shell{display:flex;gap:.2rem;background:#fff;border:1px solid var(--line);border-radius:14px;padding:.35rem;
box-shadow:var(--shadow);overflow-x:auto;scrollbar-width:none}
.nav-shell::-webkit-scrollbar{display:none}
a.nav-link{padding:.5rem 1rem;border-radius:10px;color:var(--ink)!important;font-weight:600;font-size:.92rem;white-space:nowrap;
text-decoration:none!important;transition:background .15s,color .15s}
a.nav-link:hover{background:#E5F4F2;color:#0F6D66!important}
a.to-top{position:fixed;right:1.4rem;bottom:1.4rem;z-index:1000;background:var(--teal);color:#fff!important;font-weight:700;
font-size:.85rem;padding:.65rem 1.05rem;border-radius:999px;text-decoration:none!important;box-shadow:0 8px 20px rgba(22,135,127,.35)}

.sec-head{margin:3rem 0 .6rem;padding-top:1.5rem;border-top:1px solid var(--line);scroll-margin-top:9rem}
.sec-head.first{margin-top:1.2rem;padding-top:0;border-top:0}
.eyebrow{display:inline-flex;align-items:center;gap:.5rem;font-size:.72rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;
color:#0F6D66;background:#E5F4F2;border-radius:999px;padding:.28rem .8rem .28rem .65rem}
.eyebrow span{width:7px;height:7px;border-radius:50%;background:var(--teal)}
.sec-title{font-size:1.7rem;font-weight:800;letter-spacing:-.02em;line-height:1.2;color:var(--ink);margin:.6rem 0 .3rem}
.sec-sub{font-size:1rem;color:var(--muted);max-width:780px;line-height:1.55}
.sub-head{display:flex;align-items:center;gap:.8rem;margin:1.4rem 0 .15rem;font-size:.95rem;font-weight:700;color:var(--ink)}
.sub-head::after{content:"";flex:1;height:1px;background:var(--line)}
.sub-head small{font-weight:500;color:var(--muted);font-size:.84rem}

.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:1rem;margin:.7rem 0 .3rem}
.kpi{position:relative;overflow:hidden;background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.25rem 1.3rem 1.1rem;box-shadow:var(--shadow)}
.kpi::before{content:"";position:absolute;left:1.3rem;top:0;width:34px;height:4px;border-radius:0 0 4px 4px;background:var(--accent)}
.kpi-label,.ins-label{font-size:.72rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem}
.kpi-value{font-size:2.15rem;font-weight:800;line-height:1.1;letter-spacing:-.02em;color:var(--ink)}
.kpi-note{font-size:.88rem;color:var(--muted);margin-top:.35rem;line-height:1.4}
.t-teal{--accent:var(--teal)}.t-red{--accent:var(--red)}.t-blue{--accent:var(--blue)}.t-amber{--accent:var(--amber)}

.ins-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem;margin:.8rem 0 .4rem}
.ins{position:relative;overflow:hidden;background:#fff;border:1px solid var(--line);border-radius:16px;
padding:1.25rem 1.3rem 1.3rem 1.55rem;box-shadow:var(--shadow)}
.ins::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--accent)}
.ins-stat{font-size:1.85rem;font-weight:800;line-height:1.15;letter-spacing:-.015em;color:var(--ink)}
.ins-text{font-size:.93rem;color:#475569;line-height:1.5;margin-top:.45rem}
.ins-action{margin-top:.9rem;padding-top:.8rem;border-top:1px dashed var(--line);font-size:.92rem;color:var(--ink);line-height:1.5}
.ins-action b{display:block;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--teal);margin-bottom:.25rem}

div[class*="st-key-card-"]{background:#fff;border:1px solid var(--line)!important;border-radius:16px!important;
padding:1.15rem 1.3rem 1.05rem!important;box-shadow:var(--shadow);gap:.5rem}
.card-head{display:flex;justify-content:space-between;align-items:center;gap:.75rem;flex-wrap:wrap;
padding-bottom:.7rem;margin-bottom:.4rem;border-bottom:1px solid #EEF2F6}
.card-title{font-size:1.02rem;font-weight:700;color:var(--ink)}
.chip{font-size:.74rem;font-weight:700;color:#0F6D66;background:#E5F4F2;border-radius:999px;padding:.22rem .7rem;white-space:nowrap}
.card-cap{font-size:.82rem;color:var(--muted);line-height:1.45;padding-top:.35rem}

[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.side-brand{display:flex;align-items:center;gap:.65rem;font-weight:800;font-size:1.05rem;color:var(--ink);margin:.2rem 0 .6rem}
.side-brand b{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;color:#fff;
background:linear-gradient(135deg,#136A6C,#16877F)}
.side-note{font-size:.8rem;color:var(--muted);line-height:1.5;margin-top:.4rem}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{font-weight:700;font-size:.84rem;color:var(--ink)}
.stDownloadButton button,.stButton button,.stFormSubmitButton button{border-radius:10px;font-weight:600}
[data-testid="stExpander"] details{background:#fff;border:1px solid var(--line)!important;border-radius:14px!important}
.footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.82rem;color:var(--muted);line-height:1.55}

@media(max-width:860px){
.hero-grid{grid-template-columns:1fr}}
@media(max-width:640px){
[data-testid="stMainBlockContainer"],.block-container{padding:4.2rem 1rem 3rem}
.hero{padding:1.2rem 1.1rem}.hero-title{font-size:1.6rem}.sec-title{font-size:1.4rem}.kpi-value{font-size:1.85rem}
.ins-grid{grid-template-columns:1fr}
a.to-top{right:.8rem;bottom:.8rem}}
</style>
"""

NAV_ITEMS = (
    ("overview", "Overview"),
    ("profile", "Patient profile"),
    ("outcomes", "Outcomes"),
    ("followup", "Follow-up"),
    ("statistics", "Data quality"),
    ("probability", "Statistics & Probability"),
    ("insights", "Insights"),
)


def _esc(text) -> str:
    return html.escape(str(text))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------- #
# UI building blocks
# --------------------------------------------------------------------------- #
def _render_hero(meta=None):
    stats = ""
    if meta:
        mort = meta["clean_dead"] / meta["clean_rows"] * 100
        items = (
            (f"{meta['raw_rows']:,}", "Raw records"),
            (f"{meta['clean_rows']:,}", "Patients analysed"),
            (f"{meta['outliers_removed']:,}", "Outliers removed"),
            (f"{mort:.1f}%", "Recorded mortality"),
        )
        cells = "".join(f'<div class="hero-stat"><b>{_esc(v)}</b><span>{_esc(t)}</span></div>' for v, t in items)
        stats = f'<div class="hero-stats">{cells}</div>'
    tech = "".join(f"<span>{t}</span>" for t in ("Python", "pandas", "Plotly", "SciPy", "Streamlit"))
    st.markdown(
        f'<div class="hero" id="top"><div class="hero-grid{"" if meta else " solo"}">'
        '<div class="hero-main">'
        '<div class="hero-badge"><i></i>Breast cancer analytics \u00b7 SEER-style data</div>'
        '<div class="hero-title">Breast Cancer <em>Patient Outcomes</em></div>'
        '<div class="hero-sub">Explore patient records: who the patients are, how outcomes change with stage, '
        'grade and hormone status, and what basic statistics say. Scroll down or use the menu below.</div>'
        f'<div class="hero-tech">{tech}</div></div>'
        f'{stats}</div></div>',
        unsafe_allow_html=True)


def _render_nav():
    links = "".join(f'<a class="nav-link" href="#{a}" target="_self">{_esc(t)}</a>' for a, t in NAV_ITEMS)
    st.markdown(f'<div class="nav-shell">{links}</div><a class="to-top" href="#top" target="_self">\u2191 Top</a>',
                unsafe_allow_html=True)


def _section(anchor, number, eyebrow, title, subtitle, first=False):
    st.markdown(
        f'<div class="sec-head{" first" if first else ""}" id="{anchor}"><div class="eyebrow"><span></span>{number} \u00b7 {_esc(eyebrow)}</div>'
        f'<div class="sec-title">{_esc(title)}</div><div class="sec-sub">{_esc(subtitle)}</div></div>',
        unsafe_allow_html=True)


def _sub(title, note=""):
    note_html = f"<small>{_esc(note)}</small>" if note else ""
    st.markdown(f'<div class="sub-head">{_esc(title)} {note_html}</div>', unsafe_allow_html=True)


def _kpi_grid(items):
    cells = "".join(
        f'<div class="kpi t-{tone}"><div class="kpi-label">{_esc(label)}</div><div class="kpi-value">{_esc(value)}</div>'
        f'<div class="kpi-note">{_esc(note)}</div></div>' for label, value, note, tone in items)
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def _card_header(title, chip=None):
    chip_html = f'<span class="chip">{_esc(chip)}</span>' if chip else ""
    st.markdown(f'<div class="card-head"><div class="card-title">{_esc(title)}</div>{chip_html}</div>', unsafe_allow_html=True)


def _caption(text):
    if text:
        st.markdown(f'<div class="card-cap">{_esc(text)}</div>', unsafe_allow_html=True)


def _chart_card(key, card):
    with st.container(border=True, key=f"card-{_slug(key)}"):
        _card_header(card["title"], card.get("chip"))
        st.plotly_chart(card["fig"], use_container_width=True, config=CHART_CONFIG, key=f"plot-{_slug(key)}")
        _caption(card.get("caption"))


def _chart_grid(cards, keys, cols=2):
    for i in range(0, len(keys), cols):
        for slot, key in zip(st.columns(cols, gap="medium"), keys[i:i + cols]):
            with slot:
                _chart_card(key, cards[key])


def _table_card(key, title, df, chip=None, caption=None):
    with st.container(border=True, key=f"card-{_slug(key)}"):
        _card_header(title, chip)
        st.dataframe(df, hide_index=True, use_container_width=True)
        _caption(caption)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _render_overview(view):
    _section("overview", "01", "Overview", "Patient snapshot",
             "Size, recorded outcomes and the first look at how mortality changes with stage.", first=True)
    k = view["kpis"]
    _kpi_grid([
        ("Patients in view", f"{k['n']:,}", f"{k['n'] / view['total']:.0%} of all cleaned patients", "teal"),
        ("Recorded deaths", f"{k['deaths']:,}", "Status = Dead", "red"),
        ("Observed mortality", f"{k['rate']:.1f}%", "Unadjusted share", "amber"),
        ("Median follow-up", f"{k['median_fu']:.0f} months", "Recorded survival months", "blue"),
    ])
    _chart_grid(view["cards"], ["status", "mort:6th Stage"])


def _render_profile(view):
    _section("profile", "02", "Patient profile", "Who is in the data",
             "Distributions and counts for demographics, tumor characteristics and receptor status.")
    groups = (
        ("Demographics", "Age, race and marital status",
         ["dist:Age", "count:Age_Group", "count:Race", "count:Marital Status"]),
        ("Tumor characteristics", "Size, stage and grade",
         ["dist:Tumor Size", "count:Tumor_Category", "count:T Stage", "count:N Stage", "count:6th Stage",
          "count:A Stage", "count:Grade", "count:Differentiate"]),
        ("Lymph nodes and hormone receptors", "Node involvement and ER/PR status",
         ["dist:Regional Node Examined", "dist:Regional Node Positive", "dist:Lymph_Node_Ratio",
          "count:Estrogen Status", "count:Progesterone Status", "count:Hormone_Profile"]),
    )
    for title, note, keys in groups:
        _sub(title, note)
        _chart_grid(view["cards"], keys)


def _render_outcomes(view):
    _section("outcomes", "03", "Outcomes", "Mortality across subgroups",
             "Observed (unadjusted) mortality per subgroup, compared with the overall average.")
    _sub("Mortality rate by subgroup", "Red = above overall average")
    _chart_grid(view["cards"], ["mort:Grade", "mort:Hormone_Profile", "mort:N Stage", "mort:T Stage", "mort:Age_Group",
                                "mort:Tumor_Category", "mort:Race", "mort:Marital Status", "mort:Differentiate",
                                "mort:A Stage"])
    _sub("Clinical measures by outcome", "Alive vs Dead")
    _chart_grid(view["cards"], ["violin:Tumor Size", "violin:Regional Node Positive", "violin:Lymph_Node_Ratio", "violin:Age"])
    _chart_grid(view["cards"], ["box:stage-tumor"], cols=1)


def _render_followup(view):
    m = view["metrics"]
    h = m["horizon"]
    _section("followup", "04", "Follow-up", f"Recorded follow-up and the {h}-month milestone",
             "How many patients fall in each follow-up year, and how the observed proportion without a recorded death evolves.")
    after = m["mortality_after_horizon"]
    _kpi_grid([
        (f"Observed survival at {h} mo", f"{m['survival_at_horizon']:.1f}%",
         f"Share of all patients with no recorded death by month {h}", "teal"),
        (f"Deaths within {h} mo", f"{m['deaths_by_horizon_share']:.1f}%" if m["deaths"] else "n/a",
         f"Share of all recorded deaths that happened by month {h}", "red"),
        (f"Mortality after {h} mo", f"{after:.2f}%" if not np.isnan(after) else "n/a",
         f"Share of the {m['beyond_patients']:,} patients followed beyond {h} mo who died", "blue"),
    ])
    _chart_grid(view["cards"], ["yearly", "survival"])


def _reconciliation(meta):
    rows = []
    for label, expected, key in NOTEBOOK_BASELINE:
        rows.append({"Metric": label, "Notebook": expected, "This file": int(meta[key]),
                     "Check": "\u2713 Match" if int(meta[key]) == expected else "Differs"})
    return pd.DataFrame(rows)


def _render_statistics(view, audit, meta, exports):
    _section("statistics", "05", "Data quality", "Associations and data quality",
             "Correlations with recorded outcome, plus the cleaning audit and a check against the notebook.")
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        _chart_card("heatmap", view["cards"]["heatmap"])
    with c2:
        _table_card("pb", "Point-biserial: status vs numeric", view["pb"], chip="Dead = 1",
                    caption="Association, not causation. Survival months is partly a consequence of how outcome is recorded.")
    _sub("Data quality", "Always computed on the full file")
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        _table_card("recon", "Notebook reconciliation", _reconciliation(meta),
                    caption="Compares this CSV with the notebook baseline counts.")
    with d2:
        _table_card("audit", "IQR outlier audit (1.5×, each column on full data)", audit.round(2),
            caption="Rows removed per column. A row can fail more than one column, so these counts overlap and add up to more than the total removed.")
    full_csv, audit_csv = exports
    with st.container(border=True, key="card-exports"):
        _card_header("Exports")
        b1, b2, b3 = st.columns(3, gap="small")
        b1.download_button("Cleaned data (CSV)", full_csv, "breast_cancer_cleaned.csv", "text/csv", use_container_width=True)
        b2.download_button("Current view (CSV)", view["view_csv"], "breast_cancer_view.csv", "text/csv", use_container_width=True)
        b3.download_button("IQR audit (CSV)", audit_csv, "iqr_audit.csv", "text/csv", use_container_width=True)


def _render_insights(view):
    _section("insights", "07", "Insights", "Key findings and recommendations",
             "Auto-generated from the current view. Each card shows what the data says and a suggested next step.")
    cells = "".join(
        f'<div class="ins t-{i["tone"]}"><div class="ins-label">{_esc(i["label"])}</div>'
        f'<div class="ins-stat">{_esc(i["stat"])}</div><div class="ins-text">{_esc(i["text"])}</div>'
        f'<div class="ins-action"><b>Suggested next step</b>{_esc(i["action"])}</div></div>'
        for i in view["insights"])
    st.markdown(f'<div class="ins-grid">{cells}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="card-nextsteps"):
        _card_header("Recommended next steps for deeper analysis")
        st.markdown(
            "1. **Use Kaplan-Meier curves** instead of the crude proportion; they handle patients still being followed.\n"
            "2. **Fit a Cox regression** (stage, grade, hormone profile, age together) to get adjusted risk, "
            "because the rates here are unadjusted.\n"
            "3. **Be careful with small groups:** report their size and uncertainty; validate findings with more suitable data.\n"
            "4. **Validate with clinicians** and independent data before using any finding in practice.")
        _caption("Recommendations are analytical suggestions drawn from data patterns. They are not medical advice.")

    with st.expander("Beginner glossary: how to read these findings"):
        st.markdown(
            "- **Mortality**: share of patients with a recorded death.\n"
            "- **n**: number of patients in a group. Small n means unstable results.\n"
            "- **Median**: the middle value. **IQR**: the range covering the middle 50% of patients.\n"
            "- **Stage / Grade**: how far the cancer has spread / how abnormal the cells look.\n"
            "- **ER / PR**: hormone receptors. ER+/PR+ means both are positive.\n"
            "- **Unadjusted**: other factors are not controlled for, so a pattern does not prove a cause.\n"
            "- The survival curve is a **crude observed proportion**, not Kaplan-Meier, and ignores censoring.")
    st.markdown(
        '<div class="footer">Breast Cancer Patient Outcomes \u00b7 descriptive analysis of a SEER-style dataset \u00b7 '
        'not for clinical decision-making.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar and main
# --------------------------------------------------------------------------- #
def _reset_filters(races, stages, ages):
    st.session_state["f_race"], st.session_state["f_stage"], st.session_state["f_age"] = list(races), list(stages), list(ages)


def _sidebar(clean, meta):
    races = sorted(clean["Race"].unique())
    present = set(clean["6th Stage"])
    stages = [s for s in STAGE_ORDER if s in present] + sorted(present - set(STAGE_ORDER))
    ages = [a for a in AGE_GROUP_ORDER if a in set(clean["Age_Group"])]
    for key, options in (("f_race", races), ("f_stage", stages), ("f_age", ages)):
        st.session_state.setdefault(key, list(options))
    with st.sidebar:
        st.markdown('<div class="side-brand"><b>\u271a</b>Filters</div>', unsafe_allow_html=True)
        with st.form("filters", border=False):
            r = st.multiselect("Race", races, key="f_race")
            s = st.multiselect("6th stage", stages, key="f_stage")
            a = st.multiselect("Age group", ages, key="f_age")
            st.form_submit_button("Apply filters", type="primary", use_container_width=True)
        st.button("Reset filters", on_click=_reset_filters, args=(races, stages, ages), use_container_width=True)
        st.markdown(
            f'<div class="side-note">Filters change charts, KPIs and insights. Cleaning always uses the full file '
            f'({meta["raw_rows"]:,} raw \u2192 {meta["clean_rows"]:,} clean records).</div>', unsafe_allow_html=True)
    return tuple(sorted(r)), tuple(sorted(s)), tuple(sorted(a))


def main():
    st.set_page_config(page_title="Breast Cancer Patient Outcomes", page_icon="\U0001F397\ufe0f", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    path = Path(__file__).resolve().parent / CSV_NAME
    if not path.exists():
        _render_hero()
        st.error(f"`{CSV_NAME}` was not found. Place it in this folder, next to app.py:\n\n`{path.parent}`")
        return
    mtime = path.stat().st_mtime
    try:
        clean, audit, meta = load_data(str(path), mtime)
    except Exception as exc:  # noqa: BLE001 - show a readable message instead of a traceback
        _render_hero()
        st.error(f"Could not read `{CSV_NAME}`: {exc}")
        return

    races, stages, ages = _sidebar(clean, meta)
    _render_hero(meta)
    view = build_view(str(path), mtime, races, stages, ages)
    if view["n"] == 0:
        st.warning("No patients match the selected filters. Widen the selection in the sidebar and press Apply filters.")
        return
    _render_nav()
    _render_overview(view)
    _render_profile(view)
    _render_outcomes(view)
    _render_followup(view)
    _render_statistics(view, audit, meta, export_bytes(str(path), mtime))
    clean_view = apply_filters(clean, races, stages, ages)
    render_statistics_probability(clean_view)
    _render_insights(view)


if __name__ == "__main__":
    main()
