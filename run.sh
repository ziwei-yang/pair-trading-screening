#! /bin/bash
#######################################
# script for running pair trading screener daily
# Arguments:
#   DAYS - Days to annualize return and volatility to
# 	MCAP_THRESH - Stocks below this market cap is filtered out
#######################################
DAYS=$1
MCAP_THRESH=$2
DAYS="${DAYS:-246}"
MCAP_THRESH="${MCAP_THRESH:-5000000000}"

source pvm install PairTradingScreener 3.8 
source pvm use PairTradingScreener         
pip install -r requirements.txt               

mkdir -p google_sheets
mkdir -p data

python3.8 src/price_getter.py
python3.8 src/pair_screener.py $DAYS $MCAP_THRESH
