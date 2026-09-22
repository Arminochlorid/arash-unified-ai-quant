# Methodology

## Research question

The framework asks a narrower question than “can a model predict the market?”:

> Given that a named setup has appeared, do the information and regime known at
> that moment support accepting or rejecting that setup?

Each setup remains identifiable. This permits family-specific baselines and
prevents a strong family from hiding the failure of another one.

## Causality controls

### Feature timing

Features are evaluated at bar `t`. A simulated entry occurs at the open of bar
`t+1`. The prefix-invariance audit recomputes features on truncated histories
and compares them with the corresponding prefix of a full-history calculation.

### Outcome timing

Outcomes begin at the next-bar entry. Stop and target distances use information
available at the signal bar. If one OHLC bar touches both boundaries and their
intrabar order is unknowable, the event is excluded by default.

### Walk-forward purge

For a test window starting at time `T`, training rows are admitted only when
their full label interval ended before `T`. This prevents future test-period
prices from entering the training labels.

## Model design

The classifier estimates the probability of a positive outcome. The regressor
estimates expected return in R units. A setup is eligible only when both the
probability and expected-R thresholds are met.

Categorical context includes setup family, symbol, asset class, timeframe,
regime and direction. Numeric context is generated exclusively from the market
history available at the event bar.

## Evaluation

Reported fold metrics include Brier score, AUC when both classes are present,
log loss, mean absolute error and root mean squared error. Trading summaries
are produced only from out-of-sample scored events.

## What this methodology does not solve

- selection bias in the chosen markets or time periods;
- survivorship bias in an instrument universe;
- incorrect, revised or incomplete source data;
- exchange latency, queue priority and partial fills;
- market impact and capacity;
- multiple-hypothesis inflation outside the recorded experiment set;
- future profitability.

Those limits must remain visible in every interpretation of the results.
