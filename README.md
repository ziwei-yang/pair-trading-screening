# PairTradingScreener

## Introduction
- redmine issue #2189

## Installation

---
- make sure pvm works in environment

```bash
git clone git@github.com:t-tht/PairTradingScreener.git
./run.sh
```
- run.sh can be run periodically using crontab, the maximum recommended frequency is **once per day**.



## Usage

---
```
usage: 
  run.sh [DAYS] [MCAP]
```
```
usage: pair_screener.py [-h] [-c] [-m] days mcap

positional arguments:
  days             days to annualize return and volatility to
  mcap             stocks below this market cap is filtered out

optional arguments:
  -h, --help       show this help message and exit
  -c, --use_cache  use cached data from sharpe_corr_pairs.pickle & discarded.pickle
  -m, --no_mcap    turn off market cap filter

```

### config.json
- TRADING_DAYS_PER_YEAR: 
  - defaults to 246
  - Days to annualize return and volatility to.

- MARKET_CAP_THRESH
  - defaults to 5,000,000,000
  - Stocks below this market cap is filtered out.


## Output

---
- top_sharpe_corr_pairs.csv
  - contains the metrics of top ranking stock long-short pairs according to its sharpe ratio.
  - highest absolute sharpe ratio & lowest absolute sharpe ratio
- individual_SR.csv
  - contains the metrics of every stock
- raw.csv
  - contains the metrics of every stock paired with every other stock.
  - used as the raw data in query interface on google sheets interface.
- discarded.csv
  - The stocks not included in calculations and their respective reasons.
- meta.csv
  - metadata
