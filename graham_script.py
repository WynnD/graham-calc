import os
import sys
import asyncio
import errno
import pandas as pd
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
API_KEY = os.environ['API_KEY']
REDIS_PASS = os.environ['REDIS_PASS']
r = redis.Redis(host='192.168.0.201', password=REDIS_PASS)

def storeInRedis(key, dataframe):
    df_compressed = pa.serialize(dataframe).to_buffer().to_pybytes()
    res = r.set(key, df_compressed)
    if res == True:
        print(f'{key} cached')

def loadFromRedis(key):
    data = r.get(key)
    try:
        return pa.deserialize(data)
    except:
        print("No data")

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
    try:
        cap = getCap(ticker)
        avg3yrPE = get3YrAvgPE(ticker)
        currentRatio = getCR(ticker)
        dividend = getDividendYield(ticker)
        earningsPositive = isEarningsPositiveFor10Yrs(ticker)
        earningsGrowthPercent = earningsPercentGrowthInTenYears(ticker)
        priceToBook = getPriceBookRatio(ticker)
    except Exception as e:
        print(f"Failed to process {ticker}")
        traceback.print_exc()
        return None
    if not cap:
        print("missing market cap data")
        return None
    elif not avg3yrPE:
        print("missing P/E data")
        return None
    elif not dividend:
        print("missing dividend data")
        return None
    elif not priceToBook:
        print("missing P/B ratio")
        return None
    elif not currentRatio:
        print("Missing Current ratio")
        return None
    else:
        print(f"Ticker: {ticker.upper()}    mktCap: {cap}   3yr Avg P/E: {avg3yrPE}     CurrentRatio: {currentRatio}    Current Div: {dividend} P/B: {priceToBook}")

    if cap < 2000000000:
        print("Cap too small")
    elif avg3yrPE > 15:
        print("PE too high")
    elif currentRatio and currentRatio < 2:
        print("Current ratio too low")
    elif dividend == 0:
        print("No dividend")
    elif not earningsPositive:
        print("Not earnings positive")
    elif priceToBook > 1.5:
        print("price to book too large")
    elif earningsGrowthPercent < 33.3:
        print("earnings growth too low")
    else:
        print(f"We have a winner: {ticker.upper()}")
        results.append(ticker)
        r.sadd(f"results_set_{datestring}", ticker)
    return None

async def main():
    loop = asyncio.get_event_loop()
    stocklist = cfa.available_companies()
    tickers = list(stocklist.index)
    executor = ThreadPoolExecutor(max_workers=50)
    futures = []
    for ticker in tickers:
        futures.append(
            loop.run_in_executor(executor, doProcess, ticker))
    [await f for f in tqdm(asyncio.as_completed(futures), total=len(futures))]


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
for ticker in results:
    print(ticker)
loop.close()
main()
