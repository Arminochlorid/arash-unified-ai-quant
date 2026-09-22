from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from arash_unified_ai_quant import (
    AIConfig,
    FeatureConfig,
    OutcomeConfig,
    combine_markets,
    fit_final_model,
    performance_report,
    prefix_invariance_audit,
    prepare_market,
    read_ohlcv_csv,
    save_bundle,
    walk_forward_ai,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-aware multi-market setup evaluation with purged "
            "walk-forward machine learning"
        )
    )
    parser.add_argument(
        "csv",
        nargs="+",
        help="OHLCV CSV files with timestamp,open,high,low,close[,volume]",
    )
    parser.add_argument("--symbol", nargs="*", default=None)
    parser.add_argument("--asset-class", default="UNKNOWN")
    parser.add_argument("--timeframe", default="UNKNOWN")
    parser.add_argument("--model-out", default="arash_ai_model.joblib")
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--min-events", type=int, default=500)
    return parser


def run(args: argparse.Namespace) -> dict:
    feature_cfg = FeatureConfig()
    outcome_cfg = OutcomeConfig()
    ai_cfg = AIConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        min_train_events=args.min_events,
    )

    markets = {}
    audits = {}
    for index, path in enumerate(args.csv):
        symbol = (
            args.symbol[index]
            if args.symbol and index < len(args.symbol)
            else Path(path).stem
        )
        frame = read_ohlcv_csv(path)
        audit = prefix_invariance_audit(frame, feature_cfg)
        audits[symbol] = audit
        if not audit["pass"]:
            raise RuntimeError(f"Causality audit failed for {symbol}: {audit}")
        markets[symbol] = prepare_market(
            frame,
            symbol,
            args.asset_class,
            args.timeframe,
            feature_cfg,
            outcome_cfg,
        )

    event_table = combine_markets(markets)
    walk_forward = walk_forward_ai(event_table, feature_cfg, outcome_cfg, ai_cfg)
    performance = performance_report(walk_forward["scored_events"], ai_cfg)

    report = {
        "causality_audits": audits,
        "folds": walk_forward["folds"].to_dict(orient="records"),
        "AI": performance["AI"],
        "FAIL_AUDIT": performance["FAIL_AUDIT"],
        "BASELINES": performance["BASELINES"],
    }

    bundle = fit_final_model(event_table, feature_cfg, outcome_cfg, ai_cfg)
    save_bundle(bundle, args.model_out)

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, default=str))
    print(f"MODEL_SAVED {args.model_out}")
    if args.report_out:
        print(f"REPORT_SAVED {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
