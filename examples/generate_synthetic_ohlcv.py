from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int) -> pd.DataFrame:
    if rows < 500:
        raise ValueError("rows must be at least 500")

    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    state = np.zeros(rows)
    volatility = np.full(rows, 0.35)

    for position in range(1, rows):
        block = (position // 600) % 4
        if block == 0:
            drift = 0.05
        elif block == 1:
            drift = -0.05
        elif block == 2:
            drift = -0.15 * state[position - 1]
        else:
            drift = 0.0
            volatility[position] = 0.8
        state[position] = (
            0.88 * state[position - 1]
            + drift
            + rng.normal(0, volatility[position])
        )

    close = 5000 + np.cumsum(state * 0.2)
    gap = rng.normal(0, 0.03, rows)
    open_price = np.r_[close[0], close[:-1] + gap[1:]]
    spread = rng.uniform(0.05, 0.6, rows)
    high = np.maximum(open_price, close) + spread
    low = np.minimum(open_price, close) - spread
    volume = rng.integers(100, 5000, rows)

    return pd.DataFrame(
        {
            "timestamp": index,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic OHLCV data for a software smoke test"
    )
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="synthetic_ohlcv.csv")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(args.rows, args.seed)
    frame.to_csv(output, index=False)
    print(f"WROTE {len(frame)} ROWS TO {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
