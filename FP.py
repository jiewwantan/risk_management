# ------------------------- IMPORT LIBRARIES --------------------
from datetime import datetime, timedelta
import pandas as pd
#pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as pdr
import numpy as np
from matplotlib import gridspec
from pandas.stats import moments
import warnings
import random
import matplotlib.pyplot as plt
from random import randint
import fix_yahoo_finance as yf
from dateutil.relativedelta import relativedelta
import math
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

# ------------------------- GLOBAL PARAMETERS -------------------------
# Set total fund pool
INVEST_FUND = 10000

# Start and end period of historical data in question
START = datetime(2008, 7, 1)
END = datetime(2018, 7, 12)

PRE_START = datetime(2008, 1, 1)
PRE_END = datetime(2008, 6, 30)

START_IN = datetime(2008, 6, 30)
END_IN = datetime(2018, 7, 11)


# DJIA component stocks
SYMBOL = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DWDP', 'XOM', 'GE', 'GS',
          'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UTX',
          'UNH', 'VZ', 'V', 'WMT']
SYMBOL_N = [' 3M', 'American Express', 'Apple', 'Boeing', 'Caterpillar', 'Chevron', 'Cisco', 'Coca-Cola', 'Disney',
            'DowDuPont Inc', 'Exxon Mobil', 'General Electric', 'Goldman Sachs', 'Home Depot', 'IBM', 'Intel',
            'Johnson & Johnson', 'JPMorgan Chase', ' McDonalds', 'Merck', 'Microsoft', 'Nike', 'Pfizer',
            'Procter & Gamble',
            'Travelers Companies Inc', 'United Technologies', 'UnitedHealth', 'Verizon', 'Visa', 'Wal-Mart']

LIQUIDATE_THRESHOLD = 0.2
RANDOM_PORTFOLIO_STOCKS = random.sample(SYMBOL, 10)
ADD_STOCKS = [15,20,25,30]
PER_TICK_PRICE = 5
COVER_PCT = [0.5, 0.75, 1]

# ------------------------------ CLASSES ---------------------------------
class DataRetrieval:
    """
    This class prepares data by downloading historical data from pre-saved data.
    """

    def get_dailyprice_df(self):
        self.dow_stocks = pd.read_csv('dow_stocks.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)
        self.dow_stocks_pre = pd.read_csv('dow_stocks_pre.csv', index_col='Date', parse_dates=True,
                                     infer_datetime_format=True)
    def get_dowfutures(self):
        self.dowfutures = pd.read_csv('Dow30Futures.csv', index_col='Date', parse_dates=True)

    def get_dowindex(self):
        self.dowindex = pd.read_csv('dji_10Y.csv', index_col='Date', parse_dates=True)

    def get_dowvolitity(self):
        self.dowvolatility = pdr.DataReader('VXDCLS', 'fred', START_IN, END_IN, retry_count=10)

    def get_all(self):

        self.get_dowfutures()
        self.get_dowindex()
        self.get_dowvolitity()
        self.get_dailyprice_df()
        return self.dow_stocks, self.dow_stocks_pre, self.dowfutures, self.dowindex, self.dowvolatility

class Option:
    """
    This class computes option greeks & premiums using Black-scholes calculations
    """

    def __init__(self, right, s, k, eval_date, exp_date, price=None, rf=0.01, vol=0.3,
                 div=0):
        self.k = float(k)
        self.s = float(s)
        self.rf = float(rf)
        self.vol = float(vol)
        self.eval_date = eval_date
        self.exp_date = exp_date
        self.t = self.calculate_t()
        if self.t == 0: self.t = 0.000001  ## Case valuation in expiration date
        self.price = price
        self.right = right  ## 'C' or 'P'
        self.div = div

    def calculate_t(self):
        if isinstance(self.eval_date, str):
            if '/' in self.eval_date:
                (day, month, year) = self.eval_date.split('/')
            else:
                (day, month, year) = self.eval_date[6:8], self.eval_date[4:6], self.eval_date[0:4]
            d0 = datetime(int(year), int(month), int(day))
        elif type(self.eval_date) == float or type(self.eval_date) == long or type(self.eval_date) == np.float64:
            (day, month, year) = (str(self.eval_date)[6:8], str(self.eval_date)[4:6], str(self.eval_date)[0:4])
            d0 = datetime(int(year), int(month), int(day))
        else:
            d0 = self.eval_date

        if isinstance(self.exp_date, str):
            if '/' in self.exp_date:
                (day, month, year) = self.exp_date.split('/')
            else:
                (day, month, year) = self.exp_date[6:8], self.exp_date[4:6], self.exp_date[0:4]
            d1 = datetime(int(year), int(month), int(day))
        elif type(self.exp_date) == float or type(self.exp_date) == long or type(self.exp_date) == np.float64:
            (day, month, year) = (str(self.exp_date)[6:8], str(self.exp_date)[4:6], str(self.exp_date)[0:4])
            d1 = datetime(int(year), int(month), int(day))
        else:
            d1 = self.exp_date

        return (d1 - d0).days / 365.0

    def get_price_delta(self):
        d1 = (math.log(self.s / float(self.k)) + (self.rf + self.div + math.pow(self.vol, 2) / 2.0) * self.t) / float(
            self.vol * math.sqrt(self.t))
        d2 = d1 - self.vol * math.sqrt(self.t)
        if self.right == 'C':
            self.calc_price = (norm.cdf(d1) * self.s * math.exp(-self.div * self.t) - norm.cdf(d2) * self.k * math.exp(
                -self.rf * self.t))
            self.delta = norm.cdf(d1)
        elif self.right == 'P':
            self.calc_price = (
                    -norm.cdf(-d1) * self.s * math.exp(-self.div * self.t) + norm.cdf(-d2) * self.k * math.exp(
                -self.rf * self.t))
            self.delta = -norm.cdf(-d1)

    def get_call(self):
        d1 = (math.log(self.s / self.k) + (self.rf + math.pow(self.vol, 2) / 2.0) * self.t) / (
                self.vol * math.sqrt(self.t))
        d2 = d1 - self.vol * math.sqrt(self.t)
        self.call = (norm.cdf(d1) * self.s - norm.cdf(d2) * self.k * math.exp(-self.rf * self.t))
        # put =  ( -norm.cdf(-d1) * self.s + norm.cdf(-d2) * self.k * math.exp( -self.rf * self.t ) )
        self.call_delta = norm.cdf(d1)

    def get_put(self):
        d1 = (math.log(self.s / self.k) + (self.rf + math.pow(self.vol, 2) / 2) * self.t) / (
                self.vol * math.sqrt(self.t))
        d2 = d1 - self.vol * math.sqrt(self.t)
        # call = ( norm.cdf(d1) * self.s - norm.cdf(d2) * self.k * math.exp( -self.rf * self.t ) )
        self.put = (-norm.cdf(-d1) * self.s + norm.cdf(-d2) * self.k * math.exp(-self.rf * self.t))
        self.put_delta = -norm.cdf(-d1)

    def get_theta(self, dt=0.0027777):
        self.t += dt
        self.get_price_delta()
        after_price = self.calc_price
        self.t -= dt
        self.get_price_delta()
        orig_price = self.calc_price
        self.theta = (after_price - orig_price) * (-1)

    def get_gamma(self, ds=0.01):
        self.s += ds
        self.get_price_delta()
        after_delta = self.delta
        self.s -= ds
        self.get_price_delta()
        orig_delta = self.delta
        self.gamma = (after_delta - orig_delta) / ds

    def get_all(self):
        self.get_price_delta()
        self.get_theta()
        self.get_gamma()
        return self.calc_price, self.delta, self.theta, self.gamma

    def get_impl_vol(self):
        """
        This function will iterate until finding the implied volatility
        """
        ITERATIONS = 100
        ACCURACY = 0.05
        low_vol = 0
        high_vol = 1
        self.vol = 0.5
        self.get_price_delta()
        for i in range(ITERATIONS):
            if self.calc_price > self.price + ACCURACY:
                high_vol = self.vol
            elif self.calc_price < self.price - ACCURACY:
                low_vol = self.vol
            else:
                break
            self.vol = low_vol + (high_vol - low_vol) / 2.0
            self.get_price_delta()

        return self.vol


class MathCalc:
    """
    This class performs all the mathematical calculations
    """

    @staticmethod
    def calc_return(period):
        """
        This function compute the return of a series
        """
        period_return = period / period.shift(1) - 1
        return period_return[1:len(period_return)]

    @staticmethod
    def calc_monthly_return(series):
        """
        This function computes the monthly return

        """
        return MathCalc.calc_return(series.resample('M').last())

    @staticmethod
    def positive_pct(series):
        """
        This function calculate the probably of positive values from a series of values.
        :param series:
        :return:
        """
        return float(len(series[series > 0])) / float(len(series))

    @staticmethod
    def calc_yearly_return(series):
        """
        This function computes the yearly return

        """
        return MathCalc.calc_return(series.resample('AS').last())

    @staticmethod
    def max_drawdown(r):
        """
        This function calculates maximum drawdown occurs in a series of cummulative returns
        """
        dd = r.div(r.cummax()).sub(1)
        maxdd = dd.min()
        return round(maxdd, 2)

    @staticmethod
    def calc_lake_ratio(series):

        """
        This function computes lake ratio

        """
        water = 0
        earth = 0
        series = series.dropna()
        water_level = []
        for i, s in enumerate(series):
            if i == 0:
                peak = s
            else:
                peak = np.max(series[0:i])
            water_level.append(peak)
            if s < peak:
                water = water + peak - s
            earth = earth + s
        return water / earth

    @staticmethod
    def calc_gain_to_pain(returns):
        """
        This function computes the gain to pain ratio given a series of profits and losees

        """
        profit_loss = np.array(returns)
        sum_returns = returns.sum()
        sum_neg_months = abs(returns[returns < 0].sum())
        gain_to_pain = sum_returns / sum_neg_months

        # print "Gain to Pain ratio: ", gain_to_pain
        return gain_to_pain

    @staticmethod
    def calc_kpi(portfolio):
        """
        This function calculates individual portfolio KPI related its risk profile
        """

        kpi = pd.DataFrame(index=['KPI'],columns=['Avg. monthly return', 'Pos months pct', 'Avg yearly return',
                                                  'Max monthly dd','Max dd', 'Lake ratio', 'Gain to Pain'])
        kpi['Avg. monthly return'].iloc[0] = MathCalc.calc_monthly_return(portfolio['Values']).mean()*100
        kpi['Pos months pct'].iloc[0] = MathCalc.positive_pct(portfolio['Returns'])
        kpi['Avg yearly return'].iloc[0] = MathCalc.calc_yearly_return(portfolio['Values']).mean()*100
        kpi['Max monthly dd'].iloc[0] = MathCalc.max_drawdown(MathCalc.calc_monthly_return(portfolio['CumReturns']))
        kpi['Max dd'].iloc[0] = MathCalc.max_drawdown(MathCalc.calc_return(portfolio['CumReturns']))
        kpi['Lake ratio'].iloc[0] = MathCalc.calc_lake_ratio(portfolio['CumReturns'])
        kpi['Gain to Pain'].iloc[0] = MathCalc.calc_gain_to_pain(portfolio['Returns'])

        return kpi

    @staticmethod
    def assemble_cum_returns(portfolio_longonly, portfolio_stoploss, hedged_portfolio, portfolio_15, portfolio_20,
                             portfolio_25, portfolio_30, portfolio_nc_15, portfolio_nc_20, portfolio_nc_25,
                             portfolio_nc_30, portfolio_lgcc_50, portfolio_lgcc_75, portfolio_lgcc_100):

        """
        This function assembles cumulative returns of all portfolios.
        """
        cum_returns = pd.DataFrame()
        cum_returns['Long Only 10'] = portfolio_longonly
        cum_returns['Stop Loss 10'] = portfolio_stoploss
        cum_returns['Futures Hedged 10'] = hedged_portfolio
        cum_returns['Diversified 15'] = portfolio_15
        cum_returns['Diversified 20'] = portfolio_20
        cum_returns['Diversified 25'] = portfolio_25
        cum_returns['Diversified 30'] = portfolio_30
        cum_returns['Non-correlate 15'] = portfolio_nc_15
        cum_returns['Non-correlate 20'] = portfolio_nc_20
        cum_returns['Non-correlate 25'] = portfolio_nc_25
        cum_returns['Non-correlate 30'] = portfolio_nc_30
        cum_returns['50% Covered call 30'] = portfolio_lgcc_50
        cum_returns['75% Covered call 30'] = portfolio_lgcc_75
        cum_returns['100% Covered call 30'] = portfolio_lgcc_100

        return cum_returns

    @staticmethod
    def assemble_returns(portfolio_longonly, portfolio_stoploss, hedged_portfolio, portfolio_15, portfolio_20,
                             portfolio_25, portfolio_30, portfolio_nc_15, portfolio_nc_20, portfolio_nc_25,
                             portfolio_nc_30, portfolio_lgcc_50, portfolio_lgcc_75, portfolio_lgcc_100):

        """
        This function assembles returns of all portfolios.
        """
        returns = pd.DataFrame()
        returns['Long Only 10'] = portfolio_longonly
        returns['Stop Loss 10'] = portfolio_stoploss
        returns['Futures Hedged 10'] = hedged_portfolio
        returns['Diversified 15'] = portfolio_15
        returns['Diversified 20'] = portfolio_20
        returns['Diversified 25'] = portfolio_25
        returns['Diversified 30'] = portfolio_30
        returns['Non-correlate 15'] = portfolio_nc_15
        returns['Non-correlate 20'] = portfolio_nc_20
        returns['Non-correlate 25'] = portfolio_nc_25
        returns['Non-correlate 30'] = portfolio_nc_30
        returns['50% Covered call 30'] = portfolio_lgcc_50
        returns['75% Covered call 30'] = portfolio_lgcc_75
        returns['100% Covered call 30'] = portfolio_lgcc_100

        return returns

    @staticmethod
    def colrow(i):
        """
        This function calculate the row and columns index number based on the total number of subplots in the plot.

        Return:
             row: axis's row index number
             col: axis's column index number
        """

        # Do odd/even check to get col index number
        if i % 2 == 0:
            col = 0
        else:
            col = 1
        # Do floor division to get row index number
        row = i // 2

        return col, row
class Trading:
    """
    This class performs trading and all other functions related to trading
    """

    def __init__(self, dow_stocks, dow_stocks_pre, dowfutures, dowindex, dowvolatility):
        self._dow_stocks = dow_stocks
        self._dow_stocks_pre = dow_stocks_pre
        self._dowfutures = dowfutures
        self._djia = dowindex
        self._vxd = dowvolatility
        self.remaining_stocks()

    def remaining_stocks(self):
        """
        This function finds out the remaining Dow component stocks after 10 randomly chosen stocks are taken.
        :return:
        """
        dow_remaining = self._dow_stocks.drop(RANDOM_PORTFOLIO_STOCKS, axis=1)
        self.dow_remaining = [i for i in dow_remaining.columns]

    def construct_book(self, dow_stocks_values):
        """
        This function construct the trading book for stock
        """
        portfolio = pd.DataFrame(index=dow_stocks_values.index,
                                 columns=["Values", "ProfitLoss", "Returns", "CumReturns"])
        portfolio["Values"] = dow_stocks_values.sum(axis=1)
        portfolio["ProfitLoss"] = portfolio["Values"] - portfolio["Values"].shift(1).fillna(portfolio["Values"][0])
        portfolio["Returns"] = portfolio["Values"] / portfolio["Values"].shift(1) - 1
        portfolio["CumReturns"] = portfolio["Returns"].add(1).cumprod().fillna(1)
        return portfolio

    def construct_futures_book(self, futures_values):
        """
        This function construct the trading book for futures trade
        """
        portfolio = pd.DataFrame(index=futures_values.index, columns=["Values", "ProfitLoss", "CumsumPnL"])
        portfolio["Values"] = futures_values
        # Hedging P/L, multiply by $5 as mini Dow tick price, negative as it is a short position.
        portfolio["ProfitLoss"] = - PER_TICK_PRICE * (
                portfolio["Values"] - portfolio["Values"].shift(1).fillna(portfolio["Values"][0]))
        portfolio["CumsumPnL"] = portfolio["ProfitLoss"].cumsum()

        return portfolio

    def longonly_trade(self):
        """
        This function performs a long only trade on 10 randomly chosen Dow stocks on the first day of trading, hold the
        stocks until the last trading day in the window.
        """
        # Calculate equally weighted fund allocation for each stock
        single_component_fund = INVEST_FUND / 10
        share_distribution = single_component_fund / self._dow_stocks[RANDOM_PORTFOLIO_STOCKS].iloc[0]
        dow_stocks_values = self._dow_stocks[RANDOM_PORTFOLIO_STOCKS].mul(share_distribution, axis=1)
        portfolio = self.construct_book(dow_stocks_values)
        kpi = MathCalc.calc_kpi(portfolio)
        return portfolio, kpi

    def liquidate_checker(self, today, stock, dow_stocks):
        """
        This function checks if it is necessary to liquidate the stock from the portfolio
        It will return True if it found that the current stock price drops more than 20% from the last 6 months high.
        """
        sixmonth_ago = today - relativedelta(months=+6)
        # Check if the calculated six month ago date falls on a trading day
        # If not, add one day to it until it does
        while sixmonth_ago not in dow_stocks.index:
            sixmonth_ago = sixmonth_ago + timedelta(days=1)
        sixmonth_max = max(dow_stocks[stock].loc[sixmonth_ago.date():today.date()])
        if dow_stocks[stock].loc[today] < (1 - LIQUIDATE_THRESHOLD) * sixmonth_max:
            return True
        else:
            return False

    def stoploss_trade(self):
        """
        This function performs long only trades with a portfolio of 10 randomly chosen stocks, it liquidate the stock
        if the stock drop more than 20% from the high of the past 6 months.
        """
        single_component_fund = INVEST_FUND / 10
        dow_stocks_full = pd.concat([self._dow_stocks_pre, self._dow_stocks], axis=0)
        dow_stocks_values = pd.DataFrame(index=self._dow_stocks.index, columns=SYMBOL)
        self.remaining_stocks()
        drs = self.dow_remaining
        # doing this for every stock in the portfolio
        for stock in RANDOM_PORTFOLIO_STOCKS:
            # Slide through the timeline
            for d in self._dow_stocks.index:
                # if this is the first month, calculate simulated stock price based on last available stock price
                if d == self._dow_stocks.index[0]:
                    stock_quantity = single_component_fund / self._dow_stocks[stock].loc[d]
                if self.liquidate_checker(d, stock, dow_stocks_full):
                    if len(drs) >= 2:
                        # Get the liquidated fund after exited position from this asset
                        liquidated_fund = self._dow_stocks[stock].loc[d] * stock_quantity
                        # Randomly choose a new stock
                        stock = random.sample(drs, 1)
                        # Change to this chosen stock
                        stock = stock[0]
                        drs.remove(stock)
                    if len(drs) == 1:
                        # Get the liquidated fund after exited position from this asset
                        liquidated_fund = self._dow_stocks[stock].loc[d] * stock_quantity
                        # Change to this remaining stock
                        stock = drs[0]
                        drs.remove(stock)
                    # Calculate the new stock quantity for this new stock
                    stock_quantity = liquidated_fund / self._dow_stocks[stock].loc[d]
                # Record it in the stock position value book
                dow_stocks_values[stock].loc[d] = stock_quantity * self._dow_stocks[stock].loc[d]
        portfolio = self.construct_book(dow_stocks_values)
        kpi = MathCalc.calc_kpi(portfolio)

        return portfolio, kpi

    def hedged_trade(self):
        """
        This function performs trade with a portfolio of 10 randomly chosen stocks hedged with dow futures.
        """
        portfolio_fund = INVEST_FUND * 0.95
        hedging_margin = INVEST_FUND - portfolio_fund
        # Calculate equally weighted fund allocation for each stock
        single_component_fund = portfolio_fund / 10
        # Calculate the number of futures contract to enter position,
        # With per tick price considered, this futures position is equivalent to stocks portfolio value $9500
        futures_position = portfolio_fund / (self._dowfutures['Open'].iloc[0] * PER_TICK_PRICE)
        futures_values = futures_position * self._dowfutures['Price']
        hedged_port = self.construct_futures_book(futures_values)
        # Assuming margin position is closed at the close of the last day, $500 margin is returned along the P/L of the day.
        hedged_port['ProfitLoss'].iloc[-1] = hedged_port['ProfitLoss'].iloc[-1] + hedging_margin
        # Calculate individual component stocks position
        share_distribution = single_component_fund / self._dow_stocks[RANDOM_PORTFOLIO_STOCKS].iloc[0]
        # Calculate comoonent stocks position values as time goes
        dow_stocks_values = self._dow_stocks[RANDOM_PORTFOLIO_STOCKS].mul(share_distribution, axis=1)
        # Assemble all assets values (stocks + futures P/L)
        stock_n_hedge_values = pd.concat([dow_stocks_values, hedged_port['ProfitLoss']], axis=1,
                                         join_axes=[dow_stocks_values.index])
        stock_n_hedge_values['ProfitLoss'] = stock_n_hedge_values['ProfitLoss'].fillna(0).cumsum()
        portfolio = self.construct_book(stock_n_hedge_values)
        kpi = MathCalc.calc_kpi(portfolio)
        return portfolio, kpi

    def diversified_trade(self, rps, num):
        """
        This function create trading book for the diversifed portfolios
        """
        # Calculate equally weighted fund allocation for each stock
        single_component_fund = INVEST_FUND / num
        # Randomly choose the set number of stocks from DJIA pool of component stocks
        share_distribution = single_component_fund / self._dow_stocks[rps].iloc[0]
        dow_stocks_values = self._dow_stocks[rps].mul(share_distribution, axis=1)
        portfolio = self.construct_book(dow_stocks_values)
        kpi = MathCalc.calc_kpi(portfolio)
        return portfolio, kpi

    def diversified_trades(self):
        """
        This function identify new stocks to add for new portfolios and assemble portfolios of different stocks number
        diversification.
        """
        self.remaining_stocks()
        drs = self.dow_remaining
        rps = RANDOM_PORTFOLIO_STOCKS
        for a in ADD_STOCKS:
            adding_stocks = random.sample(drs, 5)
            # add stocks to the random portfolio stock
            rps = rps + adding_stocks
            # Remove the added stocks from the remaining stock list
            drs = [x for x in drs if x not in adding_stocks]
            vars()['portfolio_' + str(a)], vars()['kpi_' + str(a)] = self.diversified_trade(rps, a)
        return vars()['portfolio_15'], vars()['kpi_15'], vars()['portfolio_20'], vars()['kpi_20'], vars()[
            'portfolio_25'], vars()['kpi_25'], vars()['portfolio_30'], vars()['kpi_30']

    def stocks_corr(self, portfolio_longonly_pre):
        """
        This function calculate the correlation coefficient between a portfolio returns and a stock returns
        """

        remaining_corr = pd.Series(index=self.dow_remaining)
        for stock in self.dow_remaining:
            stock_return = MathCalc.calc_return(self._dow_stocks_pre[stock])
            remaining_corr[stock] = portfolio_longonly_pre['Returns'][1:].corr(stock_return)
        return remaining_corr.sort_values(ascending=True)

    def non_correlate_trades(self):
        """
        This function performs trade with a portfolio starting with 10 randomly chosen stocks, creating new portfolios
        each 5 more additional stock with the stocks with less correlation with the 10 stock portfolio chosen first.
        """

        single_component_fund = INVEST_FUND / 10
        share_distribution = single_component_fund / self._dow_stocks_pre[RANDOM_PORTFOLIO_STOCKS].iloc[0]
        dow_stocks_values = self._dow_stocks_pre[RANDOM_PORTFOLIO_STOCKS].mul(share_distribution, axis=1)
        portfolio_longonly_pre = self.construct_book(dow_stocks_values)

        remaining_corr = self.stocks_corr(portfolio_longonly_pre)
        lower = 0
        upper = 5
        rps = RANDOM_PORTFOLIO_STOCKS
        self.remaining_stocks()
        drs = self.dow_remaining
        for a in ADD_STOCKS:
            adding_stocks = [i for i in remaining_corr[lower:upper].index]
            lower += 5
            upper += 5
            # add stocks to the random portfolio stock
            rps = rps + adding_stocks
            # Remove the added stocks from the remaining stock list
            drs = [x for x in drs if x not in adding_stocks]
            vars()['portfolio_nc_' + str(a)], vars()['kpi_nc_' + str(a)] = self.diversified_trade(rps, a)
        return vars()['portfolio_nc_15'], vars()['kpi_nc_15'], vars()['portfolio_nc_20'], vars()['kpi_nc_20'], vars()[
            'portfolio_nc_25'], vars()['kpi_nc_25'], vars()['portfolio_nc_30'], vars()['kpi_nc_30']

    def longonly_coveredcall_trade(self, pct):
        """
        This function performs long only trading with all 30 Dow stocks coupled with covered call trade,
        selling call options contract.
        """

        sell_call_days = pd.date_range(start=START, end=END, freq='2W-TUE')
        # risk free rate, 3-month treasury Yield
        rf = 0.0197
        # Dividen
        div = 0.0
        # Call option
        right = 'C'
        option_greeks = pd.DataFrame(
            columns=['Premium', 'Theta', 'Volatility', 's', 'k', 'eval', 'exp', 'Contract_sold', 'Premium Values'])
        for day in sell_call_days:
            while day not in self._djia.index:
                day = day + timedelta(days=1)
            # Proxy volatility with Cboe DJIA Volatility Index, VXD
            vol = self._vxd.loc[day].VXDCLS / 100
            # Current underlying price
            s = self._djia.loc[day].Close / 100
            # Strike price is 3 strikes out-of-money, +3.5 is approximately 3-4 strikes away
            k = round(s, 0) + 3.5
            # Current date when option is transacted
            eval_date = day.strftime('%Y%m%d')
            # Expiry date
            exp_date = (day + relativedelta(weeks=+2)).strftime('%Y%m%d')
            opt_contract = Option(s=s, k=k, eval_date=eval_date, exp_date=exp_date, rf=rf, vol=vol, right=right,
                                  div=div)
            premium, delta, theta, gamma = opt_contract.get_all()
            contract_sold = INVEST_FUND / (k * 100) * PER_TICK_PRICE * pct
            premium_value = contract_sold * premium * 100
            option_greeks.loc[day] = premium, theta, vol, s, k, eval_date, exp_date, contract_sold, premium_value

        # Calculate equally weighted fund allocation for each stock,
        # This is the same as the 30 diversified stocks portfolio in def diversified_trades()
        single_component_fund = INVEST_FUND / 30
        share_distribution = single_component_fund / self._dow_stocks.iloc[0]
        dow_stocks_values = self._dow_stocks.mul(share_distribution, axis=1)
        stock_n_covered_values = pd.concat([dow_stocks_values, option_greeks['Premium Values']], axis=1,
                                           join_axes=[dow_stocks_values.index])
        stock_n_covered_values['Premium Values'] = stock_n_covered_values['Premium Values'].fillna(0).cumsum()
        portfolio = self.construct_book(stock_n_covered_values)
        kpi = MathCalc.calc_kpi(portfolio)
        return portfolio, kpi

class UserInterfaceDisplay:
    """
    The class to display plot(s) to users
    """
    def plot_portfolio_return(self, cum_returns):
        """
        Function to plot all portfolio cumulative returns
        """
        # Set a palette so that all 14 lines can be better differentiated
        color_palette = ['#36C4FE', '#FF66F9', '#FF7E66', '#DE00BD', '#DE0049', '#DE0A00', '#626D00', '0038E7',
                         '#758CFF', '#4400E7', '#A2ED00', '#00EDC3', '#EECF00', '#EE5C00']
        fig, ax = plt.subplots(figsize=(14, 6))

        # Iterate the compared list to get correlation coefficient array for every compared index
        # Plot the correlation line on the plot canvas
        for i, d in enumerate(cum_returns):
            ax.plot(cum_returns.index, cum_returns[d], '-', label=cum_returns.columns[i], linewidth=2.5,
                    color=color_palette[i])

        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Cumulative returns')
        plt.title('Cumulative returns for portfolios with different risk management strategies')
        # Display and save the graph
        plt.savefig('portfolios_returns.png')
        # Inform user graph is saved and the program is ending.
        print(
        "Plot saved as portfolios_returns.png. When done viewing, please close this plot for next plot. Thank You!")

        plt.show()

    def plot_portfolio_risk(self, returns):
        """
        This function plot the histograms of returns for all portfolios.
        """

        plt.close('all')
        # Define axes, number of rows and columns
        f, ax = plt.subplots(7, 2, figsize=(16, 24))
        plt.subplots_adjust(hspace=0.5)

        for i, d in enumerate(returns):
            # Do odd/even check to col number for plot axes
            col, row = MathCalc.colrow(i)

            # plot line graph
            ax[row, col].hist(returns[d], bins=30, color='darkgreen')
            ax[row, col].axvline(returns[d].mean(), color='red',
                                 linestyle='-.', linewidth=2.5, label='Mean')
            ax[row, col].axvline(np.median(returns[d]), color='#f1f442',
                                 linestyle='-.', linewidth=2.5, label='Median')
            ax[row, col].axvline(np.median(returns[d]) + returns[d].std(), color='#b2f441', linestyle='--', linewidth=2,
                                 label='1 x sigma')
            ax[row, col].axvline(np.median(returns[d]) - returns[d].std(),
                                 color='#b2f441', linestyle='--', linewidth=2)
            ax[row, col].set_title("Returns histogram for portfolio {}".format(returns.columns[i]), fontsize=14)
            ax[row, col].legend()

        plt.savefig('portfolios_risk.png')

        print(
            "Plot saved as portfolios_risk.png. When done viewing, please close this plot to end program. Thank You!")

        plt.show()
    # ----------------------------- MAIN PROGRAM ---------------------------------

def main():
    """
    The main program

    """
    print ("\n")
    print ("################### Implementing portfolios with varied risk management strategies  ######################")
    print ("\n")

    # Set the dataframe to display values using non-scientific format, all their columns and set the canvas width
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', 7)
    pd.set_option('display.width', 1000)
    dow_stocks, dow_stocks_pre, dowfutures, dowindex, dowvolatility = DataRetrieval().get_all()
    portfolios = Trading(dow_stocks, dow_stocks_pre, dowfutures, dowindex, dowvolatility)
    portfolio_longonly, kpi_longonly = portfolios.longonly_trade()

    print ("Long only portfolio diversified with 10 Dow component stocks's KPI>")
    print (kpi_longonly)
    print ("\n")

    portfolio_stoploss, kpi_stoploss = portfolios.stoploss_trade()

    print ("Long portfolio with stop loss portfolio's KPI>")
    print (kpi_stoploss)
    print ("\n")

    hedged_portfolio, hedged_kpi = portfolios.hedged_trade()

    print ("Long portfolio hedged with Dow futures's KPI>")
    print (hedged_kpi)
    print ("\n")

    portfolio_15, kpi_15, portfolio_20, kpi_20, portfolio_25, kpi_25, portfolio_30, kpi_30 \
        = portfolios.diversified_trades()

    print ("Long portfolio diversified with 15 Dow component stocks' KPI>")
    print (kpi_15)
    print ("\n")
    print ("Long portfolio diversified with 20 Dow component stocks' KPI>")
    print (kpi_20)
    print ("\n")
    print ("Long portfolio diversified with 25 Dow component stocks' KPI>")
    print (kpi_25)
    print ("\n")
    print ("Long portfolio diversified with 30 Dow component stocks' KPI>")
    print (kpi_30)
    print ("\n")

    portfolio_nc_15, kpi_nc_15, portfolio_nc_20, kpi_nc_20, portfolio_nc_25, kpi_nc_25, portfolio_nc_30, kpi_nc_30 \
        = portfolios.non_correlate_trades()

    print ("Long portfolio diversified with 15 non-correlation preferred Dow component stocks' KPI>")
    print (kpi_nc_15)
    print ("\n")
    print ("Long portfolio diversified with 20 non-correlation preferred Dow component stocks' KPI>")
    print (kpi_nc_20)
    print ("\n")
    print ("Long portfolio diversified with 25 non-correlation preferred Dow component stocks' KPI>")
    print (kpi_nc_25)
    print ("\n")
    print ("Long portfolio diversified with 30 non-correlation preferred Dow component stocks' KPI>")
    print (kpi_nc_30)
    print ("\n")

    portfolio_lgcc_50, kpi_lgcc_50 = portfolios.longonly_coveredcall_trade(COVER_PCT[0])
    portfolio_lgcc_75, kpi_lgcc_75 = portfolios.longonly_coveredcall_trade(COVER_PCT[1])
    portfolio_lgcc_100, kpi_lgcc_100 = portfolios.longonly_coveredcall_trade(COVER_PCT[2])

    print ("Long portfolio with all Dow component stocks and 50% covered call's KPI>")
    print (kpi_lgcc_50)
    print ("\n")
    print ("Long portfolio with all Dow component stocks and 75% covered call's KPI>")
    print (kpi_lgcc_75)
    print ("\n")
    print ("Long portfolio with all Dow component stocks and 100% covered call's KPI>")
    print (kpi_lgcc_100)
    print ("\n")

    cum_returns = MathCalc.assemble_cum_returns(portfolio_longonly['CumReturns'], portfolio_stoploss['CumReturns'],
                                       hedged_portfolio['CumReturns'], portfolio_15['CumReturns'],
                                       portfolio_20['CumReturns'], portfolio_25['CumReturns'],
                                       portfolio_30['CumReturns'], portfolio_nc_15['CumReturns'],
                                       portfolio_nc_20['CumReturns'], portfolio_nc_25['CumReturns'],
                                       portfolio_nc_30['CumReturns'], portfolio_lgcc_50['CumReturns'],
                                       portfolio_lgcc_75['CumReturns'], portfolio_lgcc_100['CumReturns'])

    UserInterfaceDisplay().plot_portfolio_return(cum_returns)

    returns = MathCalc.assemble_returns(portfolio_longonly['Returns'], portfolio_stoploss['Returns'],
                                                hedged_portfolio['Returns'], portfolio_15['Returns'],
                                                portfolio_20['Returns'], portfolio_25['Returns'],
                                                portfolio_30['Returns'], portfolio_nc_15['Returns'],
                                                portfolio_nc_20['Returns'], portfolio_nc_25['Returns'],
                                                portfolio_nc_30['Returns'], portfolio_lgcc_50['Returns'],
                                                portfolio_lgcc_75['Returns'], portfolio_lgcc_100['Returns'])

    UserInterfaceDisplay().plot_portfolio_risk(returns[1:])
    print ("\n")
    print ("#######################################   END OF PROGRAM   ###############################################")

if __name__ == '__main__':
    main()

    # -------------------------------- END  ---------------------------------------
