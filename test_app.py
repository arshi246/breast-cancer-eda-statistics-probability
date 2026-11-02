import numpy as np
import pandas as pd
import pytest

from app import apply_filters, observed_survival_curve, point_biserial_summary, prepare_dataset
from statistics_section import (
    _binomial_figure,
    _chi_square_result,
    _describe_data,
    _distribution_figure,
    _hypothesis_results,
    _p_text,
    _probability_table,
    _regression_result,
    _sampling_figure,
    _test_result,
    _wilson_interval,
)


def make_raw(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": rng.integers(30, 70, n),
        "Race": rng.choice(["White", "Black", "Other"], n),
        "Marital Status": rng.choice(["Married", "Single"], n),
        "T Stage ": rng.choice(["T1", "T2", "T3"], n),
        "N Stage": rng.choice(["N1", "N2", "N3"], n),
        "6th Stage": rng.choice(["IIA", "IIB", "IIIA"], n),
        "differentiate": rng.choice(["Well differentiated", "Poorly differentiated"], n),
        "Grade": rng.choice(["1", "2", "3", " anaplastic; Grade IV"], n),
        "A Stage": rng.choice(["Regional", "Distant"], n),
        "Tumor Size": rng.integers(5, 60, n),
        "Estrogen Status": rng.choice(["Positive", "Negative"], n),
        "Progesterone Status": rng.choice(["Positive", "Negative"], n),
        "Regional Node Examined": rng.integers(5, 25, n),
        "Reginol Node Positive": rng.integers(0, 6, n),
        "Survival Months": rng.integers(2, 100, n),
        "Status": rng.choice(["Alive", "Dead"], n, p=[0.8, 0.2]),
    })
    outlier = df.iloc[[1]].copy()
    outlier["Tumor Size"] = 400
    return pd.concat([df, df.iloc[[0]], outlier], ignore_index=True)


@pytest.fixture(scope="module")
def prepared():
    return prepare_dataset(make_raw())


# --------------------------------------------------------------------------- #
# Existing tests: data pipeline
# --------------------------------------------------------------------------- #
def test_prepare_cleans_and_engineers(prepared):
    cleaned, audit, meta = prepared
    assert meta["duplicates_removed"] == 1
    assert meta["outliers_removed"] >= 1
    assert 400 not in cleaned["Tumor Size"].to_numpy()
    assert meta["clean_rows"] == len(cleaned) == meta["clean_alive"] + meta["clean_dead"]
    assert set(cleaned["Grade"]) <= {"Grade 1", "Grade 2", "Grade 3", "Grade 4 (Anaplastic)"}
    for col in ("Lymph_Node_Ratio", "Survival_Years", "Age_Group", "Tumor_Category",
                "Hormone_Profile", "Survival_Year_Bracket", "Status_Binary", "Regional Node Positive", "Differentiate"):
        assert col in cleaned.columns
    assert len(audit) == 5


def test_missing_column_raises():
    with pytest.raises(ValueError):
        prepare_dataset(make_raw().drop(columns=["Age"]))


def test_apply_filters(prepared):
    cleaned = prepared[0]
    ages = cleaned["Age_Group"].dropna().unique().tolist()
    assert apply_filters(cleaned, [], cleaned["6th Stage"].unique(), ages).empty
    out = apply_filters(cleaned, ["White"], cleaned["6th Stage"].unique(), ages)
    assert set(out["Race"]) == {"White"}


def test_survival_curve(prepared):
    curve, metrics = observed_survival_curve(prepared[0])
    assert curve["Survival (%)"].iloc[0] == pytest.approx(100.0)
    assert curve["Survival (%)"].is_monotonic_decreasing
    assert curve["Month"].iloc[-1] == 60
    assert metrics["survival_at_horizon"] == pytest.approx(curve["Survival (%)"].iloc[-1])


def test_point_biserial(prepared):
    out = point_biserial_summary(prepared[0])
    assert len(out) == 6
    assert out["r"].abs().max() <= 1


# --------------------------------------------------------------------------- #
# Statistics and probability section: small helpers
# --------------------------------------------------------------------------- #
def test_wilson_interval_known_values():
    lower, upper = _wilson_interval(50, 100)
    assert lower == pytest.approx(0.4038, abs=0.002)
    assert upper == pytest.approx(0.5962, abs=0.002)
    assert lower + upper == pytest.approx(1.0, abs=1e-6)


def test_wilson_interval_edges():
    low, high = _wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)
    low, high = _wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert 0 < high < 1
    low, high = _wilson_interval(10, 10)
    assert 0 < low < 1
    assert high == pytest.approx(1.0, abs=1e-9)


def test_p_text_and_test_result():
    assert _p_text(float("nan")) == "Not available"
    assert _p_text(0.0001) == "< 0.001"
    assert _p_text(0.04567) == "0.046"
    assert "provide evidence against" in _test_result(0.01)
    assert "do not provide enough evidence" in _test_result(0.5)
    assert "Not enough information" in _test_result(float("nan"))


def test_probability_table_and_bayes():
    data = pd.DataFrame({
        "6th Stage": ["IIIC"] * 4 + ["IIA"] * 6,
        "Status_Binary": [1, 1, 0, 0] + [1, 0, 0, 0, 0, 0],
    })
    out = _probability_table(data)
    assert out["Observed percentage"].tolist() == pytest.approx([30.0, 40.0, 20.0, 50.0, 66.67], abs=0.01)
    # Bayes: P(IIIC | Dead) = P(Dead | IIIC) * P(IIIC) / P(Dead)
    assert 0.5 * 0.4 / 0.3 == pytest.approx(out["Observed percentage"].iloc[4] / 100, abs=0.001)


def test_describe_data(prepared):
    out = _describe_data(prepared[0])
    assert out["Variable"].tolist() == [
        "Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive", "Survival Months"]
    assert (out["Std. deviation"] > 0).all()
    assert (out["Patients"] == len(prepared[0])).all()
    assert out["Minimum"].le(out["Median"]).all() and out["Median"].le(out["Maximum"]).all()


# --------------------------------------------------------------------------- #
# Statistics and probability section: tests and models
# --------------------------------------------------------------------------- #
def test_hypothesis_results(prepared):
    cleaned = prepared[0]
    res = _hypothesis_results(cleaned)
    assert res["alive_n"] + res["dead_n"] == len(cleaned)
    assert 0 <= res["t_p"] <= 1
    assert res["anova_groups"] >= 2
    assert 0 <= res["anova_p"] <= 1


def test_hypothesis_single_group_gives_nan(prepared):
    alive_only = prepared[0][prepared[0]["Status"] == "Alive"]
    res = _hypothesis_results(alive_only)
    assert res["dead_n"] == 0
    assert np.isnan(res["t_p"])


def test_regression_result(prepared):
    cleaned = prepared[0]
    fig, summary = _regression_result(cleaned)
    assert summary["patients"] == len(cleaned)
    assert abs(summary["correlation"]) <= 1
    assert summary["r_squared"] == pytest.approx(summary["correlation"] ** 2)
    assert len(fig.data) == 2


def test_regression_degenerate_returns_none():
    constant = pd.DataFrame({"Tumor Size": [10] * 5, "Regional Node Positive": [1, 2, 3, 4, 5]})
    assert _regression_result(constant) == (None, None)
    tiny = pd.DataFrame({"Tumor Size": [10, 20], "Regional Node Positive": [1, 2]})
    assert _regression_result(tiny) == (None, None)


def test_chi_square_independent_table():
    rows = []
    for stage, alive, dead in (("IIA", 20, 10), ("IIB", 40, 20), ("IIIA", 20, 10)):
        rows += [(stage, "Alive")] * alive + [(stage, "Dead")] * dead
    data = pd.DataFrame(rows, columns=["6th Stage", "Status"])
    _, chi = _chi_square_result(data)
    assert chi["chi2"] == pytest.approx(0.0, abs=1e-9)
    assert chi["p_value"] == pytest.approx(1.0)
    assert chi["cramers_v"] == pytest.approx(0.0, abs=1e-6)
    assert chi["degrees_of_freedom"] == 2
    assert chi["small_expected_cells"] == 0


def test_chi_square_single_stage_returns_none(prepared):
    one_stage = prepared[0][prepared[0]["6th Stage"] == "IIA"]
    table, chi = _chi_square_result(one_stage)
    assert chi is None
    assert table.shape[0] == 1


def test_chi_square_on_prepared(prepared):
    table, chi = _chi_square_result(prepared[0])
    assert table.shape == (3, 2)
    assert 0 <= chi["p_value"] <= 1
    assert 0 <= chi["cramers_v"] <= 1
    assert chi["total_cells"] == 6


# --------------------------------------------------------------------------- #
# Statistics and probability section: figures
# --------------------------------------------------------------------------- #
def test_figures(prepared):
    cleaned = prepared[0]
    binom = _binomial_figure(cleaned)
    assert len(binom.data[0].x) == 21
    assert float(np.sum(binom.data[0].y)) == pytest.approx(1.0)

    assert len(_distribution_figure(cleaned).data) == 2

    fig, size = _sampling_figure(cleaned)
    assert size == 30
    assert len(fig.data[0].x) == 500
    assert float(np.mean(fig.data[0].x)) == pytest.approx(float(cleaned["Tumor Size"].mean()), abs=1.5)


def test_sampling_small_data(prepared):
    _, size = _sampling_figure(prepared[0].iloc[:10])
    assert size == 10


def test_distribution_constant_values():
    constant = pd.DataFrame({"Tumor Size": [10.0] * 5})
    assert len(_distribution_figure(constant).data) == 1


# --------------------------------------------------------------------------- #
# Statistics and probability section: full render, no exceptions
# --------------------------------------------------------------------------- #
def _render_script():
    import streamlit as st
    from statistics_section import render_statistics_probability

    render_statistics_probability(st.session_state["data"])


SMOKE_CASES = {
    "full": lambda d: d,
    "one_stage": lambda d: d[d["6th Stage"] == "IIA"],
    "alive_only": lambda d: d[d["Status"] == "Alive"],
    "one_row": lambda d: d.iloc[:1],
    "empty": lambda d: d.iloc[0:0],
}


@pytest.mark.parametrize("case", list(SMOKE_CASES))
def test_render_section_has_no_exception(prepared, case):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_function(_render_script, default_timeout=60)
    at.session_state["data"] = SMOKE_CASES[case](prepared[0])
    at.run()
    assert not at.exception, [e.value for e in at.exception]