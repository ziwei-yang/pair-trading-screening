#! /bin/bash

#dir="$(dirname "$0")"
#cd $dir
#source $dir/env/bin/activate
#cd $dir/src
#python3.8 price_getter.py
#python3.8 pair_screener.py 

source pvm install PairTradingScreener 3.8 
source pvm use PairTradingScreener         
pip install -r requirements.txt               

mkdir -p google_sheets
mkdir -p data

python3.8 src/price_getter.py
python3.8 src/pair_screener.py
