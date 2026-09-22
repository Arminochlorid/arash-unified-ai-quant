# Validation record

## Verified build

- Validation date: 22 September 2026
- Python: 3.12.14
- Test command: `python tests/test_engine.py`
- Result: 6 of 6 tests passed

Verified test cases:

1. feature prefix invariance;
2. separation of setup families;
3. QTE minimum-ATR enforcement;
4. next-bar labels and ambiguous-bar exclusion;
5. multi-market walk-forward evaluation and model persistence;
6. source-coverage manifest completeness.

The command-line help and module compilation were also checked successfully.

## Meaning of this result

The result establishes that the tested mechanics behave consistently for the
included deterministic synthetic cases. It does not validate market data,
establish an economic edge, reproduce real execution, or guarantee profit.

## Required before any live interpretation

1. Use independently sourced, versioned historical data.
2. Record the complete instrument universe and all attempted configurations.
3. Include realistic fees, spread, slippage and contract rolls.
4. Reserve an untouched final holdout period.
5. Conduct forward testing without changing the model after observing results.
