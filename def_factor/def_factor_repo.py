from abc import abstractmethod

# import modin.pandas as pd
import numpy as np
import pandas as pd

# import swifter
# from pandarallel import pandarallel

# pandarallel.initialize()
class Factor:
    def __init__(self, fields, prices_data):
        # self.name = name
        # self.max_window = max_window
        self.fiedls = fields
        self.prices_data = prices_data
        # self.date_bar = pd.read_pickle(
        #     r'D:\QUANT_GAME\python_game\pythonProject\DATA\local_stable_data\stock\CN_stock_data\dateBar.pkl')
    @abstractmethod
    def calc(self):
        pass


# 1.价格速度因子 v_t = (v_t/v_(t-1) - 1)
class PriceSpeed(Factor):
    # 继承factor
    def __init__(self, fields, prices_data):
        super().__init__(fields, prices_data)
        self.name = 'price_speed'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data

    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data

    def calc(self, period=1):
        price_speed = self.prepare_data['close'].groupby('stk', group_keys=False).pct_change(period)
        return price_speed


# 2.价格速度变化量因子 v_t/v_(t_1) -1
class PriceSpeedChange(Factor):
    def __init__(self, fields,prices_data):
        super().__init__(fields, prices_data)
        self.name = 'price_acceleration'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data

    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data

    def calc(self, v_period=1, a_period=1):
        price_speed = self.prepare_data['close'].groupby('stk', group_keys=False).pct_change(v_period)
        price_speed_shift_1 = price_speed.groupby('stk', group_keys=False).shift(a_period)
        price_speed_change = price_speed - price_speed_shift_1
        return price_speed_change


# 3.作用力因子 f = ma --> f = price/geo_avg_p*(r_t/r_(t-1) - 1)
class Force(Factor):
    def __init__(self, fields, prices_data):
        super().__init__(fields, prices_data)
        self.name = 'force'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data

    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data

    def calc(self, v_period=1, a_period=1):
        price = self.prepare_data['close']
        price_speed = price.groupby('stk', group_keys=False).pct_change(v_period)#.shift(-v_period)
        price_speed_shift_1 = price_speed.groupby('stk',group_keys=False).shift(a_period)
        price_speed_change = price_speed - price_speed_shift_1
        price_shift_1 = price.groupby('stk', group_keys=False).shift(1)
        geo_averge_price = (price * price_shift_1)**(1/2)
        m = geo_averge_price.copy()
        force = price/m * price_speed_change
        return force

# 4.impluse 冲量因子 mv_2 - mv_1 -> geo_avg_p*r_t - geo_avg_p*r_(t-delta_t)
class Impluse(Factor):
    def __init__(self,fields,prices_data):
        super().__init__(fields,prices_data)
        self.name = 'impluse'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data
    
    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data
    
    def calc(self,delta_period=5):
        price_t = self.prepare_data['close']
        return_t = self.prepare_data['close'].groupby('stk', group_keys=False).pct_change(1)
        return_delta_t = return_t.shift(delta_period)
        price_max = self.prepare_data['close'].groupby('stk', group_keys=False).rolling(delta_period).max()
        price_min = self.prepare_data['close'].groupby('stk', group_keys=False).rolling(delta_period).min()
        price_geo_avg = ((price_max*price_min)**(1/2)).shift(1)
        impluse = price_geo_avg.values * return_t - price_geo_avg.values * return_delta_t
        return impluse
    
# 5.最大最小价格平均因子 1/2(p_max+p_min) - (p_max * p_min)^(1/2)
class MaxMinPriceAverage(Factor):
    def __init__(self,fields,prices_data):
        super().__init__(fields,prices_data)
        self.name ='max_min_price_average'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data
    
    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data
    
    def calc(self,delta_period=5):
        #rolling 
        # factor = self.prepare_data['close'].swifter.groupby(\
        #     'stk', group_keys=False).rolling(\
        #         delta_period).apply(self._calc_stk_df)
        price_max = self.prepare_data['close'].groupby('stk', group_keys=False).rolling(delta_period).max()
        price_min = self.prepare_data['close'].groupby('stk', group_keys=False).rolling(delta_period).min()
        price_num_mean = (price_max + price_min)/2
        price_geo_mean = (price_max*price_min)**(1/2)
        factor = price_num_mean - price_geo_mean
        factor.index = self.prepare_data['close'].index
        # factor = (price_max + price_min).mean() - np.sqrt(price_max * price_min,axis=1)
        return factor
    
    # def _calc_stk_df(self,df):
    #     df_max = df.max()
    #     df_min = df.min()
    #     return np.mean(df_max+df_min) - np.sqrt(df_max*df_min)
    
# 6.周期价格平均因子 周期算术平均 - 周期几何平均
class PeriodPriceAverage(Factor):
    def __init__(self,fields,prices_data):
        super().__init__(fields,prices_data)
        self.name = 'period_price_average'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data
    
    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data
    
    def calc(self,delta_period=5):
        price_t = self.prepare_data['close']
        period_price_num_average = price_t.groupby('stk',group_keys=False).rolling(\
            delta_period).mean()
        period_price_geo_average = price_t.groupby('stk',group_keys=False).rolling(\
            delta_period).apply(lambda df :df.prod()**(1/delta_period))
        factor = period_price_num_average - period_price_geo_average
        factor.index = price_t.index
        return factor
    
    # def _calc_stk_df(self,df):
    #     # df_num_avg
    #     df_num_avg = df.mean()
    #     # df_geo_avg
    #     df_geo_avg = np.prod(df)**(1/len(df))
    #     return df_num_avg - df_geo_avg
    
# 7.电势因子，零电势p_(t-delta_t) 或 rolling(delta_t).mean()
class Elevation(Factor):
    def __init__(self,fields,prices_data):
        super().__init__(fields,prices_data)
        self.name = 'elevation'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data
    
    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data
    
    def calc(self,zero_elevation_type:str,delta_period=5):
        price_t = self.prepare_data['close']
        if zero_elevation_type == 'p(delta_t)':
            zero_elevation = price_t.groupby('stk',group_keys=False).shift(delta_period)
        if zero_elevation_type == 'mean':
            zero_elevation = price_t.groupby('stk', group_keys=False).rolling(delta_period).mean()
        factor = price_t/zero_elevation.values - 1
        factor.index = price_t.index
        return factor
    
# 8.电压因子（电势差因子）
class Voltage(Factor):
    def __init__(self,fields,prices_data):
        super().__init__(fields,prices_data)
        self.name = 'voltage'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data
    
    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data
    
    def calc(self,delta_period=5):
        price_t = self.prepare_data['close']
        zero_elevation = price_t.groupby('stk',group_keys=False).shift(delta_period)
        price_t_1 = price_t.groupby('stk',group_keys=False).shift(1) 
        factor = (price_t/zero_elevation-1) - (price_t_1/zero_elevation-1)
        factor.index = price_t.index
        return factor
    
# 9.电流因子，电势差除以电阻
class Current(Factor):
    def __init__(self,fields,prices_data):
        super().__init__(fields,prices_data)
        self.name = 'current'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data
    
    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data
    
    def calc(self,zero_elevation_type:str,delta_period=5,resistance_period=30):
        # 直接计算与零电势的电势差
        price_t = self.prepare_data['close']
        if zero_elevation_type == 'p(delta_t)':
            zero_elevation = price_t.groupby('stk',group_keys=False).shift(delta_period)
        if zero_elevation_type == 'mean':
            zero_elevation = price_t.groupby('stk', group_keys=False).rolling(delta_period).mean()
        # 计算电势差
        t = zero_elevation
        elevation_diff = price_t - zero_elevation.values
        # 计算电阻,用给定电阻周期内的回撤表示电阻
        # resistance_period_max_price = price_t.groupby('stk', group_keys=False).rolling(resistance_period).max()
        resistance_period_min_price = price_t.groupby('stk', group_keys=False).rolling(resistance_period).min()
        resistance = resistance_period_min_price.values
        factor = elevation_diff / resistance
        factor.index = price_t.index
        return factor

# 10.收益累积因子
class ReturnAccumulation(Factor):
    def __init__(self,fields,prices_data):
        super().__init__(fields,prices_data)
        self.name = 'profit_accumulation'
        self.max_window = 10
        self.fields = fields
        self.prices_data = prices_data
    
    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data

    # 乘上一组递增数列作为权重，抵消远期的收益影响
    def calc(self,delta_period=5,sequence=(1,2,3,5,8)):
        price_t = self.prepare_data['close']
        return_t = price_t.groupby('stk', group_keys=False).pct_change(1)
        # 乘上一组递增数列作为权重，抵消远期的收益影响
        weight = pd.Series(1/np.array(sequence))
        return_accumulation = return_t.groupby('stk', group_keys=False).rolling(\
            delta_period).apply(lambda df:(np.log(df+1).values).sum())
        # return_accumulation = return_t.groupby('stk', group_keys=False).rolling(delta_period).sum()
        factor = return_accumulation
        factor.index = price_t.index
        return factor
    