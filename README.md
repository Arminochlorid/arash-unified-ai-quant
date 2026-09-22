# Arash Unified AI Quant

A leakage-aware, multi-market research framework for testing distinct trading
setup families with purged walk-forward machine learning.

The project is designed around one rule: a backtest must not use information
that was unavailable when a decision would have been made. Signals are kept as
separate setup families, entries occur on the next bar, labels crossing a
walk-forward boundary are purged, and ambiguous OHLC outcomes are excluded by
default.

## What is implemented

- Separate setup families for mean reversion, trend, factor crossovers,
  EMA-based signals, QTE-style signals, BOS and CHOCH events.
- Optional ingestion of external long/short timing columns without merging
  their identities.
- Causal feature generation with a prefix-invariance audit.
- Next-bar entry and triple-barrier outcome labelling.
- Explicit treatment of bars that hit stop and target simultaneously.
- Calendar-based, purged walk-forward evaluation.
- Gradient-boosting classifier for positive-outcome probability.
- Gradient-boosting regressor for expected return in R units.
- Baseline comparison by setup family and an explicit failed-trade audit.
- Serializable model bundle and live-candidate scoring.

## Verified scope

The included test suite verifies the mechanics below on generated data:

1. prefix invariance of causal features;
2. separation of setup families;
3. enforcement of the QTE minimum-ATR rule;
4. next-bar entries and exclusion of ambiguous barriers;
5. multi-market walk-forward training and model persistence;
6. completeness of the source-coverage manifest.

These tests verify software behaviour. They do **not** prove a profitable
strategy, reliable live execution, or future returns.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quick start

Generate a deterministic demonstration dataset:

```bash
python examples/generate_synthetic_ohlcv.py --rows 6000 --output synthetic_ohlcv.csv
```

Run the research pipeline:

```bash
arash-quant synthetic_ohlcv.csv \
  --symbol SYNTH \
  --asset-class SYNTHETIC \
  --timeframe 5m \
  --train-days 8 \
  --test-days 2 \
  --min-events 80 \
  --model-out outputs/synthetic_model.joblib \
  --report-out outputs/synthetic_report.json
```

For real research, replace the generated CSV with independently validated
historical data.

## Input schema

Each CSV requires:

```text
timestamp,open,high,low,close
```

`volume` is optional but strongly recommended. Timestamps must be parseable and
are normalized to UTC.

## Run the tests

```bash
python tests/test_engine.py
```

The expected result is `ALL 6 TESTS PASSED`.

## Repository structure

```text
arash_unified_ai_quant.py   Core features, labels, models and evaluation
run_research.py             Command-line research pipeline
tests/test_engine.py        Deterministic mechanics tests
examples/                   Synthetic, non-trading demonstration data
docs/                       Methodology and validation boundaries
```

## Research boundaries

OHLCV research cannot reproduce queue position, partial fills, market impact,
exchange outages, hidden data defects, or broker-specific behaviour. The
framework also cannot establish that an instrument universe is free of
survivorship bias. These issues require separate data and execution audits.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and
[docs/VALIDATION.md](docs/VALIDATION.md) for the exact controls and limits.

## Author

Seyed Arash Yousefi — AI Architect and CEO, IT&Science GmbH, Switzerland

## Licence

Copyright © 2026 Seyed Arash Yousefi. All rights reserved. See
[LICENSE](LICENSE). No permission is granted for commercial deployment or live
trading use.
