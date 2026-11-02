"""Educational statistics and probability section for the patient outcomes dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


TEAL = "#16877F"
BLUE = "#3D6280"
RED = "#E4686B"
INK = "#17324D"
GRID = "#EDF1F6"

CHART_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def _chart_style(fig, height=330):
    """Shared chart style. No legend: each caption explains the marks."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Source Sans 3, Source Sans Pro, Segoe UI, sans-serif",
            color=INK,
            size=13,
        ),
        margin=dict(l=15, r=15, t=25, b=15),
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=False,
        automargin=True,
        linecolor="#DCE3EC",
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        automargin=True,
    )
    return fig


def _p_text(value):
    if not np.isfinite(value):
        return "Not available"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def _test_result(p_value):
    if not np.isfinite(p_value):
        return "Not enough information for this test."
    if p_value < 0.05:
        return (
            "At the 5% level, the data provide evidence against "
            "the 'no difference or no relationship' starting assumption."
        )
    return (
        "At the 5% level, the data do not provide enough evidence "
        "against the 'no difference or no relationship' starting assumption."
    )


def _wilson_interval(successes, total, confidence=0.95):
    """Wilson confidence interval for a proportion."""
    if total <= 0:
        return float("nan"), float("nan")

    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total

    centre = (p + z2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt((p * (1 - p) / total) + z2 / (4 * total * total))
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


def _describe_data(data):
    columns = [
        "Age",
        "Tumor Size",
        "Regional Node Examined",
        "Regional Node Positive",
        "Survival Months",
    ]

    rows = []
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        rows.append(
            {
                "Variable": column,
                "Patients": int(values.size),
                "Mean": values.mean(),
                "Median": values.median(),
                "Std. deviation": values.std(ddof=1),
                "Minimum": values.min(),
                "Maximum": values.max(),
            }
        )

    return pd.DataFrame(rows).round(2)


def _probability_table(data):
    n = len(data)
    dead = data["Status_Binary"].eq(1)
    deaths = int(dead.sum())

    rows = [
        {
            "Question": "P(Dead): recorded deaths among all patients",
            "Calculation": f"{deaths:,} / {n:,}",
            "Observed percentage": 100 * deaths / n,
        }
    ]

    stage = data["6th Stage"].eq("IIIC")
    stage_n = int(stage.sum())

    if stage_n:
        stage_dead = int((stage & dead).sum())
        rows.extend(
            [
                {
                    "Question": "P(IIIC): patients in stage IIIC",
                    "Calculation": f"{stage_n:,} / {n:,}",
                    "Observed percentage": 100 * stage_n / n,
                },
                {
                    "Question": "P(Dead AND IIIC): both conditions",
                    "Calculation": f"{stage_dead:,} / {n:,}",
                    "Observed percentage": 100 * stage_dead / n,
                },
                {
                    "Question": "P(Dead | IIIC): deaths among stage IIIC",
                    "Calculation": f"{stage_dead:,} / {stage_n:,}",
                    "Observed percentage": 100 * stage_dead / stage_n,
                },
            ]
        )

        if deaths:
            rows.append(
                {
                    "Question": "P(IIIC | Dead): stage IIIC among recorded deaths",
                    "Calculation": f"{stage_dead:,} / {deaths:,}",
                    "Observed percentage": 100 * stage_dead / deaths,
                }
            )

    return pd.DataFrame(rows).round({"Observed percentage": 2})


def _distribution_figure(data):
    values = pd.to_numeric(data["Tumor Size"], errors="coerce").dropna()
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            histnorm="probability density",
            nbinsx=25,
            name="Observed tumor sizes",
            marker_color=TEAL,
            opacity=0.75,
        )
    )

    mean = values.mean()
    sd = values.std(ddof=1)

    if len(values) >= 2 and np.isfinite(sd) and sd > 0:
        x = np.linspace(values.min(), values.max(), 200)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stats.norm.pdf(x, loc=mean, scale=sd),
                mode="lines",
                name="Normal curve for comparison",
                line=dict(color=RED, width=3),
            )
        )

    fig.update_xaxes(title_text="Tumor size (mm)")
    fig.update_yaxes(title_text="Density")
    return _chart_style(fig)


def _binomial_figure(data, hypothetical_n=20):
    p = data["Status_Binary"].mean()
    x = np.arange(hypothetical_n + 1)

    fig = go.Figure(
        go.Bar(
            x=x,
            y=stats.binom.pmf(x, hypothetical_n, p),
            marker_color=BLUE,
            hovertemplate=(
                "Recorded deaths: %{x}<br>"
                "Model probability: %{y:.2%}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Number of recorded deaths in 20 patients", dtick=2)
    fig.update_yaxes(title_text="Model probability", tickformat=".0%")
    return _chart_style(fig)


def _sampling_figure(data, repetitions=500):
    values = pd.to_numeric(data["Tumor Size"], errors="coerce").dropna().to_numpy()
    sample_size = min(30, len(values))
    rng = np.random.default_rng(42)

    sample_means = np.array(
        [
            rng.choice(values, size=sample_size, replace=False).mean()
            for _ in range(repetitions)
        ]
    )

    fig = go.Figure(
        go.Histogram(
            x=sample_means,
            nbinsx=25,
            marker_color=TEAL,
            hovertemplate="Sample mean: %{x:.1f} mm<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=float(values.mean()),
        line_color=RED,
        line_width=2,
        line_dash="dash",
    )
    fig.update_xaxes(title_text=f"Mean tumor size in samples of {sample_size}")
    fig.update_yaxes(title_text="Number of samples")
    return _chart_style(fig), sample_size


def _hypothesis_results(data):
    alive = pd.to_numeric(
        data.loc[data["Status"].eq("Alive"), "Tumor Size"],
        errors="coerce",
    ).dropna()

    dead = pd.to_numeric(
        data.loc[data["Status"].eq("Dead"), "Tumor Size"],
        errors="coerce",
    ).dropna()

    result = {
        "alive_n": len(alive),
        "dead_n": len(dead),
        "alive_mean": alive.mean(),
        "dead_mean": dead.mean(),
        "t_stat": float("nan"),
        "t_p": float("nan"),
        "anova_stat": float("nan"),
        "anova_p": float("nan"),
        "anova_groups": 0,
    }

    if len(alive) >= 2 and len(dead) >= 2:
        test = stats.ttest_ind(dead, alive, equal_var=False)
        result["t_stat"] = float(test.statistic)
        result["t_p"] = float(test.pvalue)

    groups = []
    for _, group in data.groupby("6th Stage", observed=True):
        values = pd.to_numeric(group["Tumor Size"], errors="coerce").dropna()
        if len(values) >= 2:
            groups.append(values.to_numpy())

    result["anova_groups"] = len(groups)

    if len(groups) >= 2:
        test = stats.f_oneway(*groups)
        result["anova_stat"] = float(test.statistic)
        result["anova_p"] = float(test.pvalue)

    return result


def _regression_result(data):
    pairs = data[["Tumor Size", "Regional Node Positive"]].dropna()
    x = pairs["Tumor Size"].to_numpy(dtype=float)
    y = pairs["Regional Node Positive"].to_numpy(dtype=float)

    if len(x) < 3 or np.ptp(x) == 0:
        return None, None

    result = stats.linregress(x, y)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Patients",
            marker=dict(color=TEAL, opacity=0.25, size=6),
            hovertemplate=(
                "Tumor: %{x:.0f} mm<br>"
                "Positive nodes: %{y:.0f}<extra></extra>"
            ),
        )
    )

    line_x = np.array([x.min(), x.max()])
    line_y = result.intercept + result.slope * line_x

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Fitted line",
            line=dict(color=RED, width=3),
        )
    )

    fig.update_xaxes(title_text="Tumor size (mm)")
    fig.update_yaxes(title_text="Positive nodes")

    summary = {
        "patients": len(x),
        "correlation": float(result.rvalue),
        "slope": float(result.slope),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
    }

    return _chart_style(fig), summary


def _chi_square_result(data):
    table = pd.crosstab(data["6th Stage"], data["Status"])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return table, None

    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    smallest_dimension = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * smallest_dimension))

    result = {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v),
        "small_expected_cells": int((expected < 5).sum()),
        "total_cells": int(expected.size),
    }

    return table, result


def render_statistics_probability(data):
    """Render the educational section using the currently filtered patients."""
    st.markdown(
        """
        <div class="sec-head" id="probability">
            <div class="eyebrow"><span></span>06 · STATISTICS &amp; PROBABILITY</div>
            <div class="sec-title">From patterns to statistical questions</div>
            <div class="sec-sub">
                Eight guided steps using the patients currently selected in the filters.
                Results are descriptive and educational, not clinical advice.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "These calculations use the patients currently selected in the sidebar. "
        "Small groups, non-random selection and the cleaning process can affect results."
    )

    if data.empty:
        st.warning("No patients are available for this section.")
        return

    # 1. Understand the data
    st.subheader("01 · Understand the data")
    st.write(
        "Mean is the average; median is the middle value. "
        "Standard deviation describes how spread out the values are."
    )
    st.dataframe(
        _describe_data(data),
        hide_index=True,
        use_container_width=True,
    )

    # 2. Probability
    st.subheader("02 · Probability")
    st.write(
        "These are observed proportions in this dataset. "
        "The symbol | means 'among patients in this group'."
    )
    st.dataframe(
        _probability_table(data),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "P(Dead | IIIC) and P(IIIC | Dead) answer different questions. "
        "Bayes' rule connects them: "
        "P(IIIC | Dead) = P(Dead | IIIC) × P(IIIC) / P(Dead), "
        "when P(Dead) is not zero."
    )

    # 3. Distributions
    st.subheader("03 · Distributions")
    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown("**Observed tumor size and a normal curve**")
        st.plotly_chart(
            _distribution_figure(data),
            use_container_width=True,
            config=CHART_CONFIG,
            key="stats-prob-normal",
        )
        st.caption(
            "Green bars = observed patients. Red line = a normal curve for "
            "comparison; it is not proof that tumor size is normally distributed."
        )

    with right:
        st.markdown("**Binomial model: an example with 20 patients**")
        st.plotly_chart(
            _binomial_figure(data),
            use_container_width=True,
            config=CHART_CONFIG,
            key="stats-prob-binomial",
        )
        st.caption(
            "This is a teaching model using the observed death proportion. "
            "It assumes independent patients with the same probability; "
            "it is not an individual prediction."
        )

    st.caption(
        "A Poisson model is used for some counts, but it should not be assumed "
        "to fit positive lymph-node counts without checking its assumptions."
    )

    # 4. Sampling
    st.subheader("04 · Sampling and confidence interval")

    deaths = int(data["Status_Binary"].sum())
    n = len(data)
    rate = deaths / n
    lower, upper = _wilson_interval(deaths, n)

    st.markdown(
        f"**Recorded mortality:** {deaths:,} / {n:,} = **{rate:.1%}**  \n"
        f"**Illustrative 95% Wilson interval:** **{lower:.1%} to {upper:.1%}**"
    )
    st.caption(
        "This interval describes sampling uncertainty under suitable sampling assumptions. "
        "It does not correct for non-random selection, missing patients, "
        "outlier removal or differences in follow-up. It is not a world-wide estimate."
    )

    if n >= 2:
        sample_fig, sample_size = _sampling_figure(data)
        st.plotly_chart(
            sample_fig,
            use_container_width=True,
            config=CHART_CONFIG,
            key="stats-prob-sampling",
        )
        st.caption(
            f"We repeatedly take 500 random samples of {sample_size} patients "
            "from the current dataset and calculate their average tumor size. "
            "The dashed line is the current dataset's average. This illustrates "
            "sampling variation; it does not prove that this dataset represents everyone."
        )
    else:
        st.caption("At least two patients are needed for the sampling illustration.")

    # 5. Hypothesis tests
    st.subheader("05 · Hypothesis tests")
    tests = _hypothesis_results(data)

    st.markdown("**Welch t-test: average tumor size in Alive vs Dead groups**")
    st.write(
        f"Alive: n={tests['alive_n']:,}, mean={tests['alive_mean']:.2f} mm. "
        f"Dead: n={tests['dead_n']:,}, mean={tests['dead_mean']:.2f} mm."
    )
    st.write(
        f"p-value: **{_p_text(tests['t_p'])}**. "
        f"{_test_result(tests['t_p'])}"
    )

    st.markdown("**ANOVA: average tumor size across 6th-stage groups**")
    st.write(
        f"Eligible stage groups: {tests['anova_groups']}. "
        f"p-value: **{_p_text(tests['anova_p'])}**. "
        f"{_test_result(tests['anova_p'])}"
    )
    st.caption(
        "The t-test compares means, not medians. ANOVA asks whether at least "
        "one group mean differs; it does not identify which group. "
        "Both require suitable assumptions; skewed data and unequal group sizes "
        "need careful checking. A p-value does not prove a claim or measure its importance."
    )

    # 6. Relationships
    st.subheader("06 · Relationships: correlation and regression")
    regression_fig, regression = _regression_result(data)

    if regression is None:
        st.warning("Not enough variation in tumor size for this regression.")
    else:
        st.plotly_chart(
            regression_fig,
            use_container_width=True,
            config=CHART_CONFIG,
            key="stats-prob-regression",
        )
        st.caption(
            "Green dots = patients. Red line = the fitted straight line."
        )
        st.write(
            f"Correlation r: **{regression['correlation']:.3f}** · "
            f"Slope: **{regression['slope']:.3f} nodes per mm** · "
            f"R²: **{regression['r_squared']:.3f}** · "
            f"p-value: **{_p_text(regression['p_value'])}**"
        )
        st.caption(
            "This simple straight-line model describes an association between "
            "tumor size and positive-node count in this view. Node counts are "
            "count data, so a straight line may not be a suitable final model. "
            "This is not a causal claim or a patient prediction."
        )

    # 7. Chi-square
    st.subheader("07 · Chi-square: stage and recorded status")
    table, chi = _chi_square_result(data)

    st.dataframe(table, use_container_width=True)

    if chi is None:
        st.warning("At least two stage groups and both status categories are needed.")
    else:
        st.write(
            f"Chi-square: **{chi['chi2']:.2f}** · "
            f"p-value: **{_p_text(chi['p_value'])}** · "
            f"Cramér's V: **{chi['cramers_v']:.3f}**"
        )
        st.write(_test_result(chi["p_value"]))

        if chi["small_expected_cells"]:
            st.warning(
                f"{chi['small_expected_cells']} of {chi['total_cells']} expected "
                "cell counts are below 5. The chi-square approximation may be "
                "unreliable; consult an appropriate alternative test."
            )

        st.caption(
            "Cramér's V describes the strength of association from 0 to 1. "
            "A relationship does not establish that stage caused the outcome."
        )

    # 8. Report
    st.subheader("08 · Final report: how to interpret the results")
    st.markdown(
        """
- **Finding:** Report the observed percentages and differences with group sizes.
- **Uncertainty:** Report the interval and test results, not just a p-value.
- **Practical importance:** Consider the size of the difference and Cramér's V.
- **Limitations:** The data are filtered and cleaned; rates are unadjusted, and
  follow-up differs between patients.
- **Next analysis:** Check test assumptions; compare results with and without
  IQR removal; consider Kaplan–Meier and an appropriate adjusted model.
- **Clinical use:** These results are educational, not recommendations for
  individual patient care.
        """
    )