import os
import sys
import asyncio
import errno
import pandas as pd
import logging
import FundamentalAnalysis as fa
import yfinance as yf
from tqdm import tqdm
from datetime import datetime
from statistics import mean
from concurrent.futures import ThreadPoolExecutor
from requests import get
import redis
import pyarrow as pa
from functools import reduce
from time import sleep
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

datestring = datetime.now().strftime("%Y_%m_%d")
logger = logging.getLogger()
logging.basicConfig(filename=f'graham_run_{datestring}.log')
API_KEY = os.environ['API_KEY']
REDIS_PASS = os.environ['REDIS_PASS']
r = redis.Redis(host='192.168.0.201', password=REDIS_PASS)

def storeInRedis(key, dataframe):
    df_compressed = pa.serialize(dataframe).to_buffer().to_pybytes()
    res = r.set(key, df_compressed)
    if res == True:
        logger.debug(f"'{key}' cached in redis")
    else:
        logger.error(f"Failed to insert '{key}' with value of size {len(df_compressed)} ")

def loadFromRedis(key):
    data = r.get(key)
    try:
        return pa.deserialize(data)
    except:
        logger.debug(f"No data for '{key}' in redis or value is not deserializable")

class CachedFA:

    def __init__(self, api_key):
        self.api_key = api_key

    def do(self, function_name, **kwargs):
        cleaned_args = dict(kwargs)
        if 'api_key' in cleaned_args.keys():
            del cleaned_args['api_key']
        key = reduce(lambda x, y : f"{x}/{y}", cleaned_args.values(), function_name) 
        result = loadFromRedis(key)
        while not isinstance(result, pd.DataFrame):
            result = getattr(fa, function_name)(**kwargs)
            if isinstance(result, pd.DataFrame):
                storeInRedis(key, result)
            else:
                sleep(5)
        return result

    def available_companies(self, **kwargs):
        return self.do('available_companies', api_key=self.api_key, **kwargs)

    def profile(self, **kwargs):
        return self.do('profile', api_key=self.api_key, **kwargs)

    def quote(self, **kwargs):
        return self.do('quote', api_key=self.api_key, **kwargs)

    def enterprise(self, **kwargs):
        return self.do('enterprise', api_key=self.api_key, **kwargs)

    def rating(self, **kwargs):
        return self.do('rating', api_key=self.api_key, **kwargs)

    def discounted_cash_flow(self, **kwargs):
        return self.do('discounted_cash_flow', api_key=self.api_key, **kwargs)

    def earnings_calendar(self, **kwargs):
        return self.do('earnings_calendar', api_key=self.api_key, **kwargs)

    def balance_sheet_statement(self, **kwargs):
        return self.do('balance_sheet_statement', api_key=self.api_key, **kwargs)

    def income_statement(self, **kwargs):
        return self.do('income_statement', api_key=self.api_key, **kwargs)
        
    def cash_flow_statement(self, **kwargs):
        return self.do('cash_flow_statement', api_key=self.api_key, **kwargs)

    def key_metrics(self, **kwargs):
        return self.do('key_metrics', api_key=self.api_key, **kwargs)

    def financial_ratios(self, **kwargs):
        return self.do('financial_ratios', api_key=self.api_key, **kwargs)

    def financial_statement_growth(self, **kwargs):
        return self.do('financial_statement_growth', api_key=self.api_key, **kwargs)

    def stock_data(self, **kwargs):
        return self.do('stock_data', api_key=self.api_key, **kwargs)
        
    def stock_data_detailed(self, **kwargs):
        return self.do('stock_data_detailed', api_key=self.api_key, **kwargs)

    def stock_dividend(self, **kwargs):
        return self.do('stock_dividend', api_key=self.api_key, **kwargs)

cfa = CachedFA(API_KEY)

def get3YearAvgEpsStartingNYearsAgo(ticker, years):
    annualIncomeStatements = cfa.income_statement(ticker=ticker, period="annual")
    threeYearsIncomeStmt = annualIncomeStatements.iloc[:,0+years:3+years]
    threeYrEps = threeYearsIncomeStmt.loc['eps']
    meanEarnings = mean(threeYrEps)
    return meanEarnings

def isEarningsPositiveFor10Yrs(ticker):
    annualIncomeStatements = cfa.income_statement(ticker=ticker, period="annual")
    past10Years = annualIncomeStatements.iloc[:,:10]
    for eps in past10Years.loc['eps']:
        if eps < 0:
            return False
    return True

def getDividendYield(ticker):
    keyMetrics = cfa.key_metrics(ticker=ticker, period="annual")
    latestYearMetrics = keyMetrics.iloc[:,0]
    return latestYearMetrics.loc['dividendYield']

def getCap(ticker):
    data = cfa.key_metrics(ticker=ticker, period="annual")
    return data.iloc[:,0]['marketCap']

def getCR(ticker):
    keyMetrics = cfa.key_metrics(ticker=ticker, period="quarter")
    latestCR = keyMetrics.iloc[:,0].loc['currentRatio']
    return latestCR

def get3YrAvgPE(ticker):
    EPS = get3YrAvgEPS(ticker)
    currentPrice = yf.Ticker(ticker).info['regularMarketPrice']
    return currentPrice/EPS

def get3YrAvgEPS(ticker):
    return get3YearAvgEpsStartingNYearsAgo(ticker, years=0)

def earningsPercentGrowthInTenYears(ticker):
    currentAvg = get3YearAvgEpsStartingNYearsAgo(ticker, years=0)
    avgTenYearsAgo = get3YearAvgEpsStartingNYearsAgo(ticker, years=10)

    if avgTenYearsAgo < 0 or currentAvg < 0:
        return 0
    
    return ((currentAvg - avgTenYearsAgo) / avgTenYearsAgo) * 100

def getPriceBookRatio(ticker):
    keyMetrics = cfa.key_metrics(ticker=ticker, period="quarter")
    mostRecentQuarterMetrics = keyMetrics.iloc[:,0]
    bookValuePerShare = mostRecentQuarterMetrics.loc['tangibleBookValuePerShare']
    currentPrice = yf.Ticker(ticker).info['regularMarketPrice']
    return currentPrice/bookValuePerShare

results = []

def doProcess(ticker):
    # get market cap and filter
    result_string = ''
    try:
        cap = getCap(ticker)
        if not cap:
            logger.debug(f"missing market cap data for ticker {ticker}")
            return None
        avg3yrPE = get3YrAvgPE(ticker)
        if not avg3yrPE:
            logger.debug(f"missing P/E data for ticker {ticker}")
            return None
        currentRatio = getCR(ticker)
        if not currentRatio:
            logger.debug(f"Missing Current ratio for ticker {ticker}")
            return None
        dividend = getDividendYield(ticker)
        if not dividend:
            logger.debug(f"missing dividend data for ticker {ticker}")
            return None
        priceToBook = getPriceBookRatio(ticker)
        if not priceToBook:
            logger.debug(f"missing P/B ratio for ticker {ticker}")
            return None
        earningsPositive = isEarningsPositiveFor10Yrs(ticker)
        earningsGrowthPercent = earningsPercentGrowthInTenYears(ticker)
        peTimesPb = priceToBook * avg3yrPE
    except Exception as e:
        logger.debug(f"Failed to process {ticker}")
        logger.debug(traceback.format_exc())
        return None
    else:
        result_string = f"""Ticker: {ticker.upper()}
mktCap: {cap}
3yr Avg P/E: {avg3yrPE:.2f}
CurrentRatio: {currentRatio:.2f}
Current Div: {dividend:.2f}%
P/B: {priceToBook:.2f}
P/B * P/E: {peTimesPb:.2f}"""

        logger.debug(result_string)
        logger.info(f"Ticker: {ticker.upper()} fundamentals successfully acquired.")

    if cap < 2000000000:
        logger.info(f"Cap too small for ticker {ticker}")
    elif priceToBook > 1.5 and peTimesPb > 22.5:
        logger.info(f"Price to book too large for ticker {ticker}")
    elif avg3yrPE > 15 or avg3yrPE < 0:
        logger.info(f"3 yr average PE does not qualify {ticker}")
    elif currentRatio and currentRatio < 2:
        logger.info(f"Current ratio too low for ticker {ticker}")
    elif dividend == 0:
        logger.info(f"No dividend for ticker {ticker}")
    elif not earningsPositive:
        logger.info(f"Not earnings positive for ticker {ticker}")

    elif earningsGrowthPercent < 33.3:
        logger.info(f"Earnings growth too low for ticker {ticker}")
    else:
        logger.info(f"""We have a winner:
{result_string}""")
        results.append(ticker)
        r.sadd(f"results_{datestring}", ticker)
        r.hset(f"results_data_{datestring}", key=ticker, value=result_string)
    return None

async def main():
    loop = asyncio.get_event_loop()
    stocklist = cfa.available_companies()
    tickers = list(stocklist.index)
    executor = ThreadPoolExecutor()
    futures = []
    for ticker in tickers:
        futures.append(
            loop.run_in_executor(executor, doProcess, ticker))
    [await f for f in tqdm(asyncio.as_completed(futures), total=len(futures))]

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
for ticker in results:
    logger.info(ticker)
loop.close()
