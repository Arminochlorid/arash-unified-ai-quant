import os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from arash_unified_ai_quant import (
    FeatureConfig, OutcomeConfig, AIConfig,
    build_features, build_setup_events, label_events, event_feature_table,
    prepare_market, combine_markets, walk_forward_ai, performance_report,
    fit_final_model, save_bundle, load_bundle, live_candidates,
    prefix_invariance_audit, source_coverage_manifest,
)


def mkdata(n=8000, seed=7, freq="5min"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    # Regime-switching process: trends + mean reverting sections + volatility bursts.
    state = np.zeros(n)
    x = np.zeros(n)
    vol = np.full(n, 0.35)
    for i in range(1, n):
        block = (i // 600) % 4
        if block == 0: drift = 0.05
        elif block == 1: drift = -0.05
        elif block == 2: drift = -0.15 * x[i-1]
        else: drift = 0.0; vol[i] = 0.8
        eps = rng.normal(0, vol[i])
        x[i] = 0.88*x[i-1] + drift + eps
    close = 5000 + np.cumsum(x*0.2)
    gap = rng.normal(0, 0.03, n)
    open_ = np.r_[close[0], close[:-1] + gap[1:]]
    spread = rng.uniform(0.05, 0.6, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(100, 5000, n)
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":volume}, index=idx)


def test_prefix_invariance():
    d = mkdata(2500)
    r = prefix_invariance_audit(d, checkpoints=6, compare_tail=15, atol=1e-8)
    assert r["pass"], r


def test_families_not_or_merged():
    d = mkdata(2500)
    f = build_features(d)
    e = build_setup_events(d, f, "TEST", "FUTURES", "5m")
    assert "setup_family" in e
    assert e["setup_family"].nunique() >= 4
    assert not (e["setup_family"] == "COMBINED").any()


def test_qte_min_atr_is_respected():
    d = mkdata(1000)
    # Huge threshold must suppress every QTE arrow.
    cfg = FeatureConfig(qte_min_atr=1e9)
    f = build_features(d, cfg)
    assert f["qte_up_signal"].sum() == 0
    assert f["qte_down_signal"].sum() == 0


def test_labels_next_bar_and_no_ambiguous_free_win():
    d = mkdata(1200)
    f = build_features(d)
    e = build_setup_events(d, f, "TEST", "FUTURES", "5m")
    lab = label_events(d, f, e, OutcomeConfig(horizon_bars=10, ambiguous_policy="exclude"))
    if len(lab):
        for _, r in lab.head(50).iterrows():
            i = int(r["bar_index"])
            assert pd.Timestamp(r["entry_time"]) == d.index[i+1]
        amb = lab[lab["outcome"] == "AMBIGUOUS"]
        if len(amb):
            assert amb["realized_r"].isna().all()


def test_multimarket_walkforward_and_persistence():
    cfg = AIConfig(train_days=8, test_days=2, min_train_events=80, min_probability=0.50, min_expected_r=-0.25, max_iter=60, min_samples_leaf=10)
    oc = OutcomeConfig(horizon_bars=8)
    m1 = prepare_market(mkdata(4500, 10), "NQ", "INDEX_FUTURE", "5m", outcome_cfg=oc)
    m2 = prepare_market(mkdata(4500, 20), "GC", "METAL_FUTURE", "5m", outcome_cfg=oc)
    table = combine_markets({"NQ":m1,"GC":m2})
    assert len(table) > cfg.min_train_events
    wf = walk_forward_ai(table, outcome_cfg=oc, ai_cfg=cfg)
    assert len(wf["folds"]) > 0
    assert len(wf["scored_events"]) > 0
    perf = performance_report(wf["scored_events"], cfg)
    assert "AI" in perf and "BASELINES" in perf and "FAIL_AUDIT" in perf

    bundle = fit_final_model(table, outcome_cfg=oc, ai_cfg=cfg)
    p = os.path.join(tempfile.gettempdir(), "arash_ai_bundle_test.joblib")
    save_bundle(bundle, p)
    b2 = load_bundle(p)
    live = live_candidates(m1["raw"], b2, "NQ", "INDEX_FUTURE", "5m")
    assert isinstance(live, pd.DataFrame)


def test_manifest_complete():
    m = source_coverage_manifest()
    for k in ["ArashArrow","QuantEngine","BOS_CHOCH","BuySell","QTEScalper","AI","RealityChecks"]:
        assert k in m


if __name__ == "__main__":
    tests = [
        test_prefix_invariance,
        test_families_not_or_merged,
        test_qte_min_atr_is_respected,
        test_labels_next_bar_and_no_ambiguous_free_win,
        test_multimarket_walkforward_and_persistence,
        test_manifest_complete,
    ]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"ALL {len(tests)} TESTS PASSED")
