# Stock Investment Recommendations

CLI tool that uses **past stock prediction performance** to decide whether to invest in each currently predicted stock, and how much capital to allocate.

## What it does

Given:

- historical predictions (`ticker`, `predicted_return`, `actual_return`)
- current predictions (`ticker`, `predicted_return`)

the tool:

1. Learns per-ticker reliability from history (hit rate + calibration + sample confidence).
2. Produces an adjusted expected return.
3. Applies investment thresholds.
4. Sizes positions with fractional Kelly and caps per-position risk.
5. Scales allocations if total exceeds 100% of capital.

## Requirements

- Python 3.10+

## Input format

### Historical CSV

Required columns:

- `ticker`
- `predicted_return` (decimal return, e.g. `0.08` = +8%)
- `actual_return` (decimal return, e.g. `-0.03` = -3%)

Example:

```csv
ticker,predicted_return,actual_return
AAPL,0.08,0.06
AAPL,0.04,-0.01
MSFT,0.05,0.07
```

### Current CSV

Required columns:

- `ticker`
- `predicted_return`

Example:

```csv
ticker,predicted_return
AAPL,0.07
MSFT,0.06
TSLA,0.10
```

## Usage

```bash
python -m src.stock_recommender \
  --historical examples/historical_predictions.csv \
  --current examples/current_predictions.csv \
  --capital 25000 \
  --min-history 8 \
  --min-confidence 0.5 \
  --min-expected-return 0.01 \
  --max-position-pct 0.2 \
  --fractional-kelly 0.5 \
  --output-csv examples/recommendations.csv
```

### Optional JSON output

```bash
python -m src.stock_recommender \
  --historical examples/historical_predictions.csv \
  --current examples/current_predictions.csv \
  --json
```

## Output fields

- `decision`: `INVEST` or `DO_NOT_INVEST`
- `allocation_pct`: fraction of total capital (e.g. `0.08` = 8%)
- `allocation_amount`: dollar amount based on `--capital`
- `predicted_return`: raw current prediction
- `expected_return`: reliability-adjusted return
- `confidence`: historical confidence score
- `rationale`: short explanation

## Testing

```bash
python -m unittest discover -s tests
```
