import csv
import datetime as dt
import os
import pickle
from collections import OrderedDict
from pprint import pprint

import akshare as ak
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

dirname = os.path.dirname(__file__)

CURRENT_DATA_DF_FILENAME = os.path.join(dirname, "../data/hk_stock_info.csv")
STOCK_PRICE_PICKLE = os.path.join(dirname, "../data/em_data.pickle")
STOCK_PRICE_FILENAME = os.path.join(dirname, "../data/em_data.csv")
META_FILENAME = os.path.join(dirname, "../google_sheets/meta.csv")

MARKET_CAP_FILENAME = os.path.join(dirname, "../data/market_cap.pickle")


class DataMnt:

    def download_stock_hk_hist(self, file_name, start_date, end_date):
        pass

    def get_close_price_df(self, file_data):
        pass

    @staticmethod
    def get_hk_stock_name_dict(lang="sc", cache=False):
        if not cache:
            current_data_df = ak.stock_hk_spot()
            current_data_df = current_data_df[['symbol', 'name', 'engname']]
            current_data_df.to_csv(CURRENT_DATA_DF_FILENAME, index=False, encoding='utf8')
        df = pd.read_csv(CURRENT_DATA_DF_FILENAME)

        if lang == "sc":
            name_list = df['name'].tolist()
        elif lang == "en":
            name_list = df['name'].tolist()
        else:
            raise ValueError

        symbol_list = [str(_).rjust(5, "0") for _ in df['symbol'].tolist()]
        stock_name_dict = dict(zip(symbol_list, name_list))
        return stock_name_dict

    @staticmethod
    def get_stock_type(symbol: str):
        def is_symbol_in_ranges(symbol, ranges: list):
            symbol_int = int(symbol)
            for _ in ranges:
                floor, ceiling = _.split("-")
                floor = int(floor)
                ceiling = int(ceiling)
                assert floor < ceiling
                if floor <= symbol_int <= ceiling:
                    return True
            else:
                return False

        # if symbol in ['08202', '08272', '08265', '01573', '08108', '00153', '00449', '00238', '01363', '00729',
        #                 '08001', '00633']:
        #     return "Black list"
        if is_symbol_in_ranges(symbol, ["02800-02849", "03000-03199"]):
            return "Exchange Traded Funds"
        if is_symbol_in_ranges(symbol, ["04000-04199"]):
            return "Hong Kong Monetary Authority Exchange Fund Notes"
        if is_symbol_in_ranges(symbol, ["04200-04299"]):
            return "HKSAR Government Bonds"
        if is_symbol_in_ranges(symbol, ["04300-04329", "04400-04599", "05000-06029", "40000-40999"]):
            return "Debt securities for professional investors only"
        if is_symbol_in_ranges(symbol, ["04600-04699"]):
            return "Professional Preference Shares"
        if is_symbol_in_ranges(symbol, ["04700-04799"]):
            return "Debt securities for the public"
        if is_symbol_in_ranges(symbol, ["06200-06299"]):
            return "Hong Kong Depositary Receipts (HDRs)"
        if is_symbol_in_ranges(symbol, ["06300-06399"]):
            return "Securities/HDRs which are restricted securities (RS) under US federal securities laws."
        if is_symbol_in_ranges(symbol, ["06750-06799"]):
            return "Bonds of Ministry of the Finance of the People’s Republic of China"
        if is_symbol_in_ranges(symbol, ["07200-07399", "07500-07599"]):
            return "Leveraged and Inverse Products"
        if is_symbol_in_ranges(symbol, ["08000-08999"]):
            return "GEM securities"
        if is_symbol_in_ranges(symbol, ["09000-09199", "09800-09849"]):
            return "Exchange Traded Funds (traded in USD)"
        if is_symbol_in_ranges(symbol, ["09200-09399", "09500-09599"]):
            return "Leveraged and Inverse Products (traded in USD)"

        return "Normal stock"


class BTXDataMnt(DataMnt):
    """
    Bitex data management
    """

    def get_btx_symbol_list(self):
        _symbol_list = requests.get("http://192.168.2.4:8080/?querytype=stocks").json()['res']['res']
        print("Total stock: ", len(_symbol_list))
        _symbol_list = [_ for _ in _symbol_list if
                        _.startswith('0') and self.get_stock_type(_) in ["Normal stock", "GEM securities"]]
        # TODO: May remove new stocks, ETF, etc.
        return _symbol_list

    def download_stock_hk_hist(self, file_name, start_date, end_date):
        _symbol_list = self.get_btx_symbol_list()
        stocks_info_map = {}
        for stock in _symbol_list:
            res = requests.get(f"http://192.168.2.4:8080/?querytype=stockhistory&ccassdateshift=0"
                               f"&startdate={start_date}&enddate={end_date}&stocklist={stock}").json()
            stocks_info_map[stock] = OrderedDict([(_['dt'], _['data']) for _ in res['res']['res'][0]['content']])
            print(stock)

        with open(file_name, "wb") as f:
            pickle.dump(stocks_info_map, f)

    def get_close_price_list(self, file, start_date, end_date):
        _stocks_info_map = None
        with open(file, "rb") as f:
            _stocks_info_map = pickle.load(f)
        _stocks_close_price_list = []
        _symbol_list = []
        for _symbol, date_info in _stocks_info_map.items():
            if self.get_stock_type(_symbol) not in ["Normal stock", "GEM securities"]:
                print(_symbol, self.get_stock_type(_symbol))
                continue
            # if start_date not in date_info or end_date not in date_info:
            #     print("Invalid date: weekend or lack data")
            #     raise ValueError
            # if date_info[start_date]["close"] == -1:
            #     print(_symbol, "Stock not publish yet")
            #     continue

            temp = []
            for date, detail in date_info.items():
                # print(date)
                if start_date <= date <= end_date:
                    if detail['close'] == -1:
                        print(_symbol, "Close price contain -1")
                        break
                    temp.append(detail['close'])

            else:  # if no break
                if len(set(temp)) <= 5:
                    print(_symbol, "Abnormal price")
                    continue
                _symbol_list.append(_symbol)
                _stocks_close_price_list.append(temp)
        return _symbol_list, _stocks_close_price_list


class EMDataMnt(DataMnt):
    """
    East Money data management
    """

    def download_stock_hk_hist(self, file_name, start_date, end_date):
        """
        Data from East Money. Format of date: YYYYMMDD
        :return:
        """
        assert file_name.endswith(".pickle")
        stock_name_dict = self.get_hk_stock_name_dict()
        _symbol_list = list(stock_name_dict.keys())
        # _symbol_list = ['00009', '00185']
        stocks_hist_map = {}
        for index, symbol in enumerate(_symbol_list):
            print(f"Downloading [{index:04d}/{len(_symbol_list)}]{symbol} {stock_name_dict[symbol]}")
            try:
                stock_hk_hist_df = ak.stock_hk_hist(symbol=symbol, start_date=start_date, end_date=end_date,
                                                    adjust="hfq")
                # print(stock_hk_hist_df)
                stocks_hist_map[symbol] = stock_hk_hist_df
            except Exception as e:
                print("[ERROR]", symbol, stock_name_dict[symbol])
                print(e)
                continue

        with open(file_name, "wb") as f:
            pickle.dump(stocks_hist_map, f)

    def get_close_price_df(self, load_file, start_date, end_date):
        """
        Load close price dataframe from pickle file
        :param load_file: pickle file
        :param start_date: YYYYMMDD or YYYY-MM-DD
        :param end_date: YYYYMMDD or YYYY-MM-DD
        :return: pd.Dataframe
        """
        # Transform "20201212" to "2020-12-12"
        if "-" not in start_date and len(start_date) == 8:
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        if "-" not in end_date and len(end_date) == 8:
            end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        _stocks_info_map = None
        with open(load_file, "rb") as f:
            _stocks_info_map = pickle.load(f)
        _stocks_close_price_df_list = []
        for _symbol, info_df in _stocks_info_map.items():
            if self.get_stock_type(_symbol) not in ["Normal stock", "GEM securities"]:
                print(_symbol, self.get_stock_type(_symbol))
                continue
            if _symbol == '00009':
                continue
            temp_df = info_df[["日期", "收盘"]]
            temp_df = temp_df.set_index("日期")
            temp_df = temp_df.rename(index={"日期": "Date"}, columns={"收盘": _symbol})
            temp_df = temp_df.loc[start_date:end_date, :]
            _stocks_close_price_df_list.append(temp_df)
        _stocks_close_price_df = pd.concat(_stocks_close_price_df_list, axis=1)
        return _stocks_close_price_df


def get_stock_history(stock, startdate, enddate):
    # apiStockHistory
    query = f"http://192.168.2.4:8080/?querytype=stockhistory&ccassdateshift=0&startdate={startdate}&enddate={enddate}&stocklist={stock}&fields=amount"
    response = requests.get(query)
    response = response.json()
    return response

def get_mcap(startdate, enddate):
    """
    gets the outstanding shares of all stocks
    Args:
        startdate:
        enddate:

    Returns:
        None
    """
    from helper import STOCKS
    mcap = {}
    for stock in STOCKS:
        query = f"http://192.168.2.4:8080/?querytype=stockhistory&ccassdateshift=0&startdate={startdate}&enddate={enddate}&stocklist={stock}&fields=amount"
        res = requests.get(query)
        res = res.json()
        try:
            data = res["res"]["res"][0]["content"]

            cap = data[-1]['data']['amount']
            time = data[-1]['dt']
            mcap["date"] = time
            mcap[stock] = cap
        except KeyError:
            continue

    with open(MARKET_CAP_FILENAME, "wb") as f:
        pickle.dump(mcap, f)


if __name__ == "__main__":
    em_data_mnt = EMDataMnt()

    datetime_format = "%Y%m%d"

    enddate = dt.datetime.now() - relativedelta(days=1)
    startdate = enddate - relativedelta(years=1)
    yesterday = enddate - relativedelta(days=0)

    enddate = enddate.strftime(datetime_format)
    startdate = startdate.strftime(datetime_format)
    yesterday = yesterday.strftime(datetime_format)

    get_mcap(yesterday, enddate)

    em_data_mnt.download_stock_hk_hist(STOCK_PRICE_PICKLE, startdate, enddate)
    df = em_data_mnt.get_close_price_df(STOCK_PRICE_PICKLE, startdate, enddate)
    df.to_csv(STOCK_PRICE_FILENAME)

    with open(META_FILENAME, "w") as f:
        writer = csv.writer(f)
        writer.writerow([startdate, enddate])
