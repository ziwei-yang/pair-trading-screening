from __future__ import annotations

import argparse
import csv
import json
import math
import os.path
import pickle
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

from helper import timethis, log
from price_getter import MARKET_CAP_FILENAME

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("days",
                    type=int,
                    help="days to annualize return and volatility to")
parser.add_argument("mcap",
                    type=int,
                    help="stocks below this market cap is filtered out")
parser.add_argument("-c", "--use_cache",
                    help="use cached data from sharpe_corr_pairs.pickle & discarded.pickle",
                    action="store_true",
                    default=False)

parser.add_argument("-m", "--no_mcap",
                    help="turn off market cap filter",
                    action="store_true",
                    default=False)
args = parser.parse_args()

TRADING_DAYS_PER_YEAR = args.days
MARKET_CAP_THRESH = args.mcap


# hardset constants, better not to change / parameterize
LOGRET_NAN_THRESH = 5
PX_NAN_THRESH = 5

top_n_per_group = 5
display_n_groups = 5
zero_display_n_groups = 30

dirname = os.path.dirname(__file__)

DISCARDED_PICKLE_FILENAME = os.path.join(dirname, "../data/discarded.pickle")
SHARPE_CORR_PAIR_FILENAME = os.path.join(dirname, "../data/sharpe_corr_pairs.pickle")
LOGRET_FILENAME = os.path.join(dirname, "../data/logret.csv")
EM_DATA_FILENAME = os.path.join(dirname, "../data/em_data.csv")

INDIVIDUAL_SR_FILENAME = os.path.join(dirname, "../google_sheets/individual_SR.csv")
TOP_SHARPE_CORR_PAIRS_FILENAME = os.path.join(dirname, "../google_sheets/top_sharpe_corr_pairs.csv")
RAW_PAIRS_FILENAME = os.path.join(dirname, "../google_sheets/raw.csv")
DISCARDED_STOCKS_FILENAME = os.path.join(dirname, "../google_sheets/discarded.csv")



def individual_sharpe_sortino():
    """
    Calculate individual stock metrics and write to file.
    """
    df = pd.read_csv(LOGRET_FILENAME, index_col=0, header=0)
    # create downside covariance
    pos_df = df.copy()
    for col in pos_df:
        pos_df[col][pos_df[col] > 0] = 0

    cols = df.columns
    stats = {}
    for col in cols:
        lr = df[col]
        mean = lr.mean()
        std = lr.std()
        drisk = pos_df[col].std()
        stats[col] = (mean, std, drisk)
    log.info(f"there are {len(stats.keys())} stocks.")

    with open(INDIVIDUAL_SR_FILENAME, "w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["stock id", "return", "volatility", "downside volatility", "annualized sharpe ratio",
             "annualized sortino ratio"])
        for stock_name, stat in stats.items():
            days = TRADING_DAYS_PER_YEAR

            mean, std, drisk = stat
            annual_mean = mean * days
            annual_std = std * np.sqrt(days)
            annual_drisk = drisk * np.sqrt(days)
            annual_sharpe = annual_mean / annual_std
            annual_sortino = annual_mean / annual_drisk

            pct_mean = (np.exp(annual_mean) - 1)
            pct_std = (pct_mean / annual_sharpe)
            pct_drisk = pct_mean / annual_sortino
            pct_annual_sharpe = pct_mean / annual_std
            pct_annual_sortino = pct_mean / annual_drisk

            writer.writerow(
                [stock_name, pct_mean, annual_std, annual_drisk, pct_annual_sharpe, pct_annual_sortino])

            # writer.writerow(
            #     [stock_name, annual_mean, annual_std, annual_drisk, annual_sharpe, annual_sortino])


def SR_query(query: str, stocks_stats: dict, return_count=10) -> list:
    """
    output highest and lowest absolute SR pair with respect to query stock
    Args:
        query: The query stock
        stocks_stats: dictionary of stocks with its stats
        return_count:

    Returns:
        results
    """
    top_n = stocks_stats[query]
    bot_n = stocks_stats[query]
    top_n = sorted(top_n, key=lambda x: x['sharpe_ratio'], reverse=True)[:return_count]
    bot_n = sorted(bot_n, key=lambda x: abs(
        x['sharpe_ratio']), reverse=False)[:return_count]

    results = []
    for result in [top_n, bot_n]:
        for metric in result:
            results.append(
                [(metric["long"]), (metric["short"]), (metric["correlation"]), (metric["sharpe_ratio"]),
                 (metric["sortino_ratio"]),
                 (metric["annualised_pct_return"]),
                 (metric["annualised_volatility"]),
                 (metric["annualised_downside_volatility"])])
        results.append([])

    return results


def SR_rankings(formatted_metrics: dict) -> tuple:
    """
    sorts, groups and reformats the metrics based on the sharpe ratio

    Args:
        formatted_metrics:

    Returns:
        high_abs_sr_rows:
        low_abs_sr_rows:

    """
    high_abs_sr_metrics = sorted(formatted_metrics.items(),
                                 key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    low_abs_sr_metrics = sorted(formatted_metrics.items(), key=lambda x: abs(
        x[1]['sharpe_ratio']), reverse=False)

    high_abs_sr_groups = defaultdict(lambda: [])

    for pair in high_abs_sr_metrics:
        pair_name, metrics = pair
        long = pair_name[0].split("-")[1]
        short = pair_name[1].split("-")[1]
        metrics["long"] = long
        metrics["short"] = short
        high_abs_sr_groups[long].append(metrics)

    low_abs_sr_groups = []
    for pair in low_abs_sr_metrics:
        pair_name, metrics = pair
        long = pair_name[0].split("-")[1]
        short = pair_name[1].split("-")[1]
        metrics["long"] = long
        metrics["short"] = short
        low_abs_sr_groups.append(metrics)

    high_abs_sr_groups = {k: v[:top_n_per_group] for k, v in high_abs_sr_groups.items()}

    high_abs_sr_rows = []
    low_abs_sr_rows = []

    for metrics in list(high_abs_sr_groups.values())[:display_n_groups]:
        for metric in metrics:
            high_abs_sr_rows.append(
                [(metric["long"]), (metric["short"]), (metric["correlation"]), (metric["sharpe_ratio"]),
                 (metric["sortino_ratio"]),
                 (metric["annualised_pct_return"]),
                 (metric["annualised_volatility"]),
                 (metric["annualised_downside_volatility"])])

    for metric in low_abs_sr_groups[:zero_display_n_groups]:
        low_abs_sr_rows.append(
            [(metric["long"]), (metric["short"]), (metric["correlation"]), (metric["sharpe_ratio"]),
             (metric["sortino_ratio"]),
             (metric["annualised_pct_return"]),
             (metric["annualised_volatility"]),
             (metric["annualised_downside_volatility"])])

    return high_abs_sr_rows, low_abs_sr_rows


def query_to_population_SR(formatted_metrics: dict) -> tuple[dict, list]:
    """
    For each stock, return a list of SR when combined with another stock in the population, sorted from highest to
    lowest in sharpe ratio.

    Args:
        formatted_metrics:
            a dictionary containing the pairwise statistics (ret, vol, sr...) of every permutation of stock pairs.
            In the format of :
            {("long_stock0", "short_stock1"): {...sr,sortino,ret,vol...}}

    Returns:
        independent_query_result:
        header:
    """
    queries = {}

    sorted_metrics = sorted(formatted_metrics.items(),
                            key=lambda x: x[1]['sharpe_ratio'], reverse=True)

    grouped_metrics = defaultdict(lambda: [])

    for pair in sorted_metrics:
        pair_name, metrics = pair
        long = pair_name[0].split("-")[1]
        short = pair_name[1].split("-")[1]
        metrics["long"] = long
        metrics["short"] = short
        grouped_metrics[long].append(metrics)

        queries[long] = 1

    independent_query_result = {}

    for query in queries.keys():
        result = SR_query(query, grouped_metrics)
        independent_query_result[query] = result

    # for writing to csv
    header = ["Long", "Short", "Correlation", "Annualized Sharpe", "Annualized Sortino", "Return", "Volatility",
              "Downside Volatility"]

    return independent_query_result, header


@timethis
def compute_data(use_cache=True, market_cap_filter=True) -> None:
    """
    Driver function for computing all metrics related to the pair trading screener.
    Args:
        use_cache: Use previously computed results
        market_cap_filter: Whether or not to filter stocks with market cap below a certain threshold

    Returns:
        None
    """
    discarded = defaultdict(lambda: [])

    if not use_cache:



        df = pd.read_csv(EM_DATA_FILENAME, index_col=0)
        px = df.copy()
        logret = pd.DataFrame()

        for col in df.columns:
            # remove stocks with pric containing more than NaN than PX_NAN_THRESH
            if df[col].isnull().sum() > PX_NAN_THRESH:
                discarded[col].append("Null price")
                continue
            else:
                logret[col] = np.log(df[col]).diff()

        logret.to_csv(LOGRET_FILENAME)

        # create downside logret
        pos_df = get_downside_logret(logret)

        correlation, covariance, downside_covariance, primary_stock_stats = get_stock_statistics(logret, pos_df,
                                                                                                 discarded)

        stock = []
        if market_cap_filter:
            with open(MARKET_CAP_FILENAME, "rb") as f:
                market_caps = pickle.load(f)
            for stock_name in primary_stock_stats.keys():
                if stock_name in market_caps and px[stock_name].mean() * market_caps[stock_name] > MARKET_CAP_THRESH:
                    stock.append(stock_name)
                else:
                    discarded[stock_name].append("Insufficient market cap")
                    continue
        else:
            stock = list(primary_stock_stats.keys())

        metrics = estimate_pairwise_timeseries_statistics(stock, primary_stock_stats, correlation, covariance,
                                                          downside_covariance)

        with open(SHARPE_CORR_PAIR_FILENAME, "wb") as f:
            pickle.dump(metrics, f)

        with open(DISCARDED_PICKLE_FILENAME, "wb") as f:
            pickle.dump(dict(discarded), f)
    else:
        with open(SHARPE_CORR_PAIR_FILENAME, "rb") as f:
            metrics = pickle.load(f)

        with open(DISCARDED_PICKLE_FILENAME, "rb") as f:
            discarded = pickle.load(f)

    # some minor reformatting for displaying
    formatted_metrics = minor_reformatting(metrics)

    # For each stock, return a list of SR when combined with another stock in the population, sorted from highest to
    # lowest in sharpe ratio.
    independent_query_result, header = query_to_population_SR(formatted_metrics)

    # sorts, groups and reformats the metrics based on the sharpe ratio
    high_SR, zero_SR = SR_rankings(formatted_metrics)

    # writing to suitable files
    with open(TOP_SHARPE_CORR_PAIRS_FILENAME, "w",
              encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(high_SR)
        writer.writerow([])
        writer.writerows(zero_SR)

    with open(RAW_PAIRS_FILENAME, "w",
              encoding="utf8") as f:
        writer = csv.writer(f)
        for query, query_result in independent_query_result.items():
            if query_result[0] and query_result[1]:
                writer.writerows(query_result)
            else:
                pass
    with open(DISCARDED_STOCKS_FILENAME, "w",
              encoding="utf8") as f:
        writer = csv.writer(f)
        rows = []
        for stock_name, v in discarded.items():
            v.insert(0, stock_name)
            rows.append(v)
        # pprint(rows)
        writer.writerows(rows)
    individual_sharpe_sortino()


def minor_reformatting(metrics: dict) -> dict:
    """
    just some minor reformatting for displaying

    Args:
        metrics:

    Returns:
        reformatted metrics
    """
    formatted_metrics = {}
    for pair, metric in metrics.items():
        if math.isnan(metric['sharpe_ratio']) or math.isnan(metric['correlation']):
            pass
        else:
            annual_sharpe = metric['sharpe_ratio']
            annual_sortino = metric['sortino_ratio']
            pair = ("L-" + pair[0], "S-" + pair[1])
            formatted_metrics[pair] = {'sharpe_ratio': annual_sharpe,
                                       'correlation': metric['correlation'],
                                       'sortino_ratio': annual_sortino,
                                       'annualised_pct_return': metric['annualised_pct_return'],
                                       'annualised_volatility': metric['annualised_volatility'],
                                       'annualised_downside_volatility': metric['annualised_downside_volatility']}
    return formatted_metrics


def estimate_pairwise_timeseries_statistics(stocks: list,
                                            primary_stock_stats: dict,
                                            correlation: pd.DataFrame,
                                            covariance: pd.DataFrame,
                                            downside_covariance: pd.DataFrame) -> dict:
    """
    get combined timeseries statistics (sharpe ratio, sortino ratio, correlation, annualised returns, annualised vol, ...)
    Args:
        stocks: the list of stocks that we want to compute the metrics of
        primary_stock_stats: mean, std, and downside of the above stocks
        correlation: correlation matrix
        covariance: covariance matrix
        downside_covariance: downside covariance matrix

    Returns:
        dict containing sharpe ratio, correlation, sortino ratio, return, volatility, downside volatility of stock pairs

    """
    metrics = {}
    for pair in permutations(stocks, 2):

        s1 = pair[0]
        s2 = pair[1]

        pct = True
        if pct:
            # calculate pairwise pct ret
            pct1 = np.exp(TRADING_DAYS_PER_YEAR * primary_stock_stats[s1][0]) - 1
            pct2 = np.exp(TRADING_DAYS_PER_YEAR * primary_stock_stats[s2][0]) - 1
            pct_combined_mean = pct1 - pct2
        else:
            # calculate pairwise log ret
            log_combined_mean = primary_stock_stats[s1][0] - primary_stock_stats[s2][0]
            pct_combined_mean = np.exp(log_combined_mean * TRADING_DAYS_PER_YEAR) - 1
        # variance is bilinear, but covariance is linear to both random var
        combined_std = np.sqrt(
            TRADING_DAYS_PER_YEAR) * np.sqrt(
            primary_stock_stats[s1][1] ** 2 + primary_stock_stats[s2][1] ** 2 - 2 * covariance[s1][s2])
        combined_dstd = np.sqrt(TRADING_DAYS_PER_YEAR) * np.sqrt(
            primary_stock_stats[s1][2] ** 2 + primary_stock_stats[s2][2] ** 2 - 2 * downside_covariance[s1][s2])
        annual_sharpe = pct_combined_mean / combined_std
        annual_sortino = pct_combined_mean / combined_dstd

        # correlation = covariance[s1][s2] / (stats[s1][1] * stats[s2][1])
        corr = correlation[s1][s2]

        sorted_pair_name = pair
        metrics[sorted_pair_name] = {"sharpe_ratio": annual_sharpe,
                                     "correlation": corr,
                                     "sortino_ratio": annual_sortino,
                                     "annualised_pct_return": pct_combined_mean,
                                     "annualised_volatility": combined_std,
                                     "annualised_downside_volatility": combined_dstd}
    return metrics


def get_stock_statistics(logret: pd.DataFrame, neg_logret: pd.DataFrame, discarded: dict) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Computes the mean, standard deviation, downside standard deviation of individual stocks.
    Also computes the pairwise correlation, covariance, and downside covariance.

    Args:
        logret: log returns of stock
        neg_logret: negative log returns only
        discarded: stocks being discarded and the respective reason

    Returns:
        pairwise correlation, pairwise covariance, pairwise downside_covariance, primary_stats

    """
    primary_stats = {}
    for col in logret.columns:
        if logret[col].isnull().sum() > LOGRET_NAN_THRESH:
            # remove stocks with logret containing more than NaN than PX_NAN_THRESH
            discarded[col].append("Null logret")
            logret.drop(col, axis=1)
            neg_logret.drop(col, axis=1)
            continue

        lr = logret[col]
        mean = lr.mean()
        std = lr.std()
        drisk = neg_logret[col].std()

        primary_stats[col] = (mean, std, drisk)
    log.info(f"there are {len(primary_stats.keys())} stocks.")
    covariance = logret.cov()
    correlation = logret.corr()
    downside_covariance = neg_logret.cov()
    return correlation, covariance, downside_covariance, primary_stats


def get_downside_logret(logret: pd.DataFrame) -> pd.DataFrame:
    """
    removes all positive elements from a log return df

    Args:
        logret:

    Returns:
        downside log return
    """
    pos_df = logret.copy()
    for col in pos_df:
        pos_df[col][pos_df[col] > 0] = 0
    return pos_df


if __name__ == "__main__":
    print(TRADING_DAYS_PER_YEAR)
    print(MARKET_CAP_THRESH)
    compute_data(use_cache=args.use_cache,
                 market_cap_filter=not args.mcap)
