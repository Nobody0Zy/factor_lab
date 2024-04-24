import os
import pickle
from abc import abstractmethod

import math
# import modin.pandas as pd
import numpy as np
import pandas as pd
# import swifter
from pandarallel import pandarallel
from scipy import stats

pandarallel.initialize()


class FactorRepo:
    def __init__(self):
        self.factor_repo_dict = {
            'price_speed': PriceSpeed,
            'price_speed_change': PriceSpeedChange,
            'force': Force,
            'impluse': Impluse,
            'max_min_price_average': MaxMinPriceAverage,
            'period_price_average': PeriodPriceAverage,
            'voltage': Voltage,
            'resistance': Resistance,
            'current': Current,
            'power': Power,
            'entropy': Entropy,
            'fractal_dimension': FractalDimension,
            'mean_price': MeanPrice,
            'gmean_return': gMeanReturn,
            'price_position': PricePosition,
            'magnetic_field':MagneticField,
            'electromagnetic_induction':ElectromagneticInduction,
            'volatility':Volatility,
            'vibration_period':VibrationPeriod,
            'chaos':Chaos
            }


class Factor:
    def __init__(self, fields, prices_data):
        self.name = None
        self.fields = fields
        self.prices_data = prices_data
        self.other_depend_factors = None
        self.calc_params = dict()
        self.save_h5_path = r'F:\factor_lab_res\prepared_data\factor_data.h5'
        self.save_factor_info_path = r'F:\factor_lab_res\prepared_data\factor_info_dict.pkl'
        # 设定初始值域
        self._init_min_value = None
        self._init_max_value = None
        self._max_value = None
        self._min_value = None
        # 建议处理方式：None
        self.process_suggest_method = None
        # 因子值
        self.factor_data = None
        
    @property
    def factor_info(self):
        factor_info = {
            'name': self.name,
            'fields': self.fields,
            'other_dIepend_factors': self.other_depend_factors,
            'calc_params': self.calc_params,
            'save_h5_path': self.save_h5_path,
            # 值域
            'value_range': [self._init_min_value, self._init_max_value],
            # 建议处理方式：None
            'process_suggest_method': self.process_suggest_method,
        }
        return factor_info

    @property
    def prepare_data(self):
        prepare_data = self.prices_data[self.fields]
        return prepare_data

    def _add_factor_info(self, factor_info):
        # 保存factor_info到pkl文件
        if not os.path.exists(self.save_factor_info_path):
            with open(self.save_factor_info_path, 'wb') as f:
                factor_info_dict = {}
                pickle.dump(factor_info_dict, f)
        else:
            with open(self.save_factor_info_path, 'rb') as f:
                factor_info_dict = pickle.load(f)
            with open(self.save_factor_info_path, 'wb') as f:
                factor_info_dict[self.name] = factor_info
                pickle.dump(factor_info_dict, f)

    @abstractmethod
    def calc(self):
        pass

    def calc_factor_and_save_factor_info(self):
        self.factor_data = self.calc()
        if isinstance(self.factor_data, pd.Series):
            self.factor_data = pd.DataFrame(self.factor_data)
        if isinstance(self.factor_data, pd.DataFrame):
            self.factor_data.columns = [self.name]
            self._min_value = self.factor_data.min().values[0]
            self._max_value = self.factor_data.max().values[0]
            factor_info = self.factor_info.copy()
            factor_info['value_range'] = [self._min_value, self._max_value]
            self._add_factor_info(factor_info)
        else:
            raise ValueError('factor_data must be a pandas DataFrame', self.factor_data)
        return self.factor_data


# 1.价格速度因子 v_t = (v_t/v_(t-1) - 1) ,Rt = (p_t/p_(t-1) - 1)
class PriceSpeed(Factor):
    # 继承factor
    def __init__(self, fields, prices_data, v_period=1):
        super().__init__(fields, prices_data)
        self.name = 'price_speed'
        self.fields = fields
        self.prices_data = prices_data
        self.v_period = v_period
        self.calc_params = {'period': self.v_period, }
        self.other_depend_factors = []
        # 设定初始值域：[-0.25,0.25]
        self._init_min_value = -0.25
        self._init_max_value = 0.25

    def calc(self):
        period = self.v_period
        close_price = self.prepare_data['close']
        price_speed = close_price.groupby('stk', group_keys=False).pct_change(period)
        # 筛选值域以内的因子值
        # price_speed = price_speed.clip(lower=self._init_min_value, upper=self._init_max_value)
        return price_speed


# 2.价格速度变化量因子 v_t/v_(t_1) -1
class PriceSpeedChange(Factor):
    def __init__(self, fields, prices_data, v_period=1, a_period=1):
        super().__init__(fields, prices_data)
        self.name = 'price_speed_change'
        self.fields = fields
        self.prices_data = prices_data
        self.v_period = v_period
        self.a_period = a_period
        self.calc_params = {'v_period': self.v_period,
                            'a_period': self.a_period,
                            }
        self.other_depend_factors = ['price_speed']
        self._init_min_value = -0.5
        self._init_max_value = 0.5

    def calc(self):
        v_period = self.v_period
        a_period = self.a_period
        price_speed_obj = PriceSpeed(self.fields, self.prices_data, v_period)
        price_speed = price_speed_obj.calc()
        price_speed_shift_1 = price_speed.groupby('stk', group_keys=False).ffill().shift(a_period)
        price_speed_change = price_speed - price_speed_shift_1
        return price_speed_change


# 3.作用力因子 f = ma --> f = price/geo_avg_p*(r_t/r_(t-1) - 1)
class Force(Factor):
    def __init__(self, fields, prices_data, v_period=1, a_period=1):
        super().__init__(fields, prices_data)
        self.name = 'force'
        self.fields = fields
        self.prices_data = prices_data
        self.v_period = v_period
        self.a_period = a_period
        self.calc_params = {'v_period': self.v_period,
                            'a_period': self.a_period, }
        self.other_depend_factors = ['price_speed_change']
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
        self.process_suggest_method = 'standardization'

    def calc(self):
        """
        如果这里m为价格，那么就需要对因子进行标准化处理，因为因子受到了价格的影响，但是价格的单位不同,
        所以此处待定
        """
        v_period = self.v_period
        a_period = self.a_period
        price_speed_change_obj = PriceSpeedChange(self.fields, self.prices_data, v_period, a_period)
        price_speed_change = price_speed_change_obj.calc()
        price = self.prepare_data['close']
        force = price * price_speed_change
        return force


# 4.impluse 冲量因子 m_2v_2 - m_1v_1 -> p_2*r_2 - p_1*r_1
class Impluse(Factor):
    def __init__(self, fields, prices_data):
        super().__init__(fields, prices_data)
        self.name = 'impluse'
        self.fields = fields
        self.prices_data = prices_data
        self.v_period = 1
        self.delta_period = 20
        self.calc_params = {'v_period': self.v_period,
                            'delta_period': self.delta_period, }
        self.other_depend_factors = ['price_speed_change']
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
        self.process_suggest_method = 'standardization'

    def calc(self):
        v_period = self.v_period
        delta_period = self.delta_period
        price_t2 = self.prepare_data['close']
        price_t1 = price_t2.groupby('stk', group_keys=False).shift(delta_period)
        price_speed_obj = PriceSpeed(self.fields, self.prices_data, v_period)
        price_speed_t2 = price_speed_obj.calc()
        price_speed_t1 = price_speed_t2.groupby('stk', group_keys=False).shift(delta_period)
        impluse = price_t2 * price_speed_t2 - price_t1 * price_speed_t1
        impluse.index = price_t2.index
        return impluse


# 5.最大最小价格平均因子 1/2(p_max+p_min) - (p_max * p_min)^(1/2)
class MaxMinPriceAverage(Factor):
    def __init__(self, fields, prices_data):
        super().__init__(fields, prices_data)
        self.name = 'max_min_price_average'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = 20
        self.calc_params = {'delta_preiod': self.delta_period}
        self.other_depend_factors = []
        self._init_min_value = 0
        self._init_max_value = np.inf

    def calc(self):
        delta_period = self.delta_period
        price_t = self.prepare_data['close']
        price_max = price_t.groupby('stk', group_keys=False).rolling(delta_period, min_periods=1).max()
        price_min = price_t.groupby('stk', group_keys=False).rolling(delta_period, min_periods=1).min()
        price_num_mean = (price_max + price_min) / 2
        price_geo_mean = (price_max * price_min) ** (1 / 2)
        factor = price_num_mean / price_geo_mean
        factor.index = self.prepare_data['close'].index
        return factor


# 6.周期价格平均因子 周期算术平均 - 周期几何平均
class PeriodPriceAverage(Factor):
    def __init__(self, fields, prices_data):
        super().__init__(fields, prices_data)
        self.name = 'period_price_average'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = 20
        self.calc_params = {'delta_preiod': self.delta_period}
        self.other_depend_factors = []
        self._init_min_value = 0
        self._init_max_value = np.inf

    def calc(self):
        delta_period = self.delta_period
        price_t = self.prepare_data['close']
        period_price_mean = price_t.groupby('stk', group_keys=False).rolling( \
            delta_period).mean()
        period_price_gmean = price_t.groupby('stk', group_keys=False).rolling( \
            delta_period).apply(lambda df: df.prod() ** (1 / len(df)))
        factor = period_price_mean / period_price_gmean
        factor.index = price_t.index
        return factor


# 7.电压因子(电势差因子)
class Voltage(Factor):
    def __init__(self, fields, prices_data, delta_period=5,calc_price_or_returns='price', voltage_zero_elevation_type='mean'):
        super().__init__(fields, prices_data)
        self.name = 'voltage'
        self.fields = fields
        self.prices_data = prices_data
        self.calc_price_or_returns = calc_price_or_returns
        self.zero_elevation_type = voltage_zero_elevation_type
        self.delta_period = delta_period
        self.calc_params = {'zero_elevation_type': self.zero_elevation_type,
                            'delta_period': self.delta_period,
                            'calc_price_or_returns': self.calc_price_or_returns,
                            }
        self.other_depend_factors = []
        self._init_min_value = -np.inf
        self._init_max_value = np.inf

    def calc(self):
        zero_elevation_type = self.zero_elevation_type
        delta_period = self.delta_period
        if self.calc_price_or_returns == 'price':
            data_t = self.prepare_data['close']
        elif self.calc_price_or_returns == 'returns':
            price_t = self.prepare_data['close']
            data_t = price_t.groupby('stk', group_keys=False).pct_change(1)
        else:
            raise ValueError(f'calc_price_or_returns must be in {["price", "returns"]}')
        
        if zero_elevation_type == 'mean':
            zero_elevation = data_t.groupby('stk', group_keys=False) \
                .rolling(delta_period).mean()
        factor = data_t - zero_elevation.values
        factor.index = data_t.index
        return factor


# 8.电阻因子
class Resistance(Factor):
    def __init__(self, fields, prices_data, delta_period=20,calc_price_or_returns='returns', resistance_type='std'):
        super().__init__(fields, prices_data)
        self.name = 'resistance'
        self.fields = fields
        self.prices_data = prices_data
        self.resistance_type_list = ['abs_max_min', 'drawdown', 'std']
        self.calc_price_or_returns = calc_price_or_returns
        self.resistance_type = resistance_type
        self.delta_period = delta_period
        self.calc_params = {'resistance_type': self.resistance_type,
                            'resistance_type_list': self.resistance_type_list,
                            'delta_period': self.delta_period}
        self.other_depend_factors = []
        self._init_min_value = -np.inf
        self._init_max_value = np.inf

    def calc(self):
        resistance_type = self.resistance_type
        delta_period = self.delta_period
        if self.calc_price_or_returns == 'price':
            data_t = self.prepare_data['close']
        elif self.calc_price_or_returns == 'returns':
            price_t = self.prepare_data['close']
            data_t = price_t.groupby('stk', group_keys=False).pct_change(1)
        else:
            raise ValueError(f'calc_price_or_returns must be in {["price", "returns"]}')
        
        if resistance_type == 'abs_max_min':
            factor = self._abs_max_min(data_t, delta_period)
        elif resistance_type == 'drawdown':
            factor = self._drawdown(data_t, delta_period)
        elif resistance_type == 'std':
            factor = self._std(data_t, delta_period)
        else:
            raise ValueError(f'resistance_type must be in {self.resistance_type_list}')
        return factor

    def _abs_max_min(self, data_t, delta_period):
        groupby_rolling_max = data_t.groupby('stk', group_keys=False) \
            .rolling(window=delta_period, min_periods=1) \
            .max()
        groupby_rolling_min = data_t.groupby('stk', group_keys=False) \
            .rolling(window=delta_period, min_periods=1) \
            .min()
        factor = (groupby_rolling_max - groupby_rolling_min).abs()
        factor.index = data_t.index
        return factor

    def _drawdown(self, data_t, delta_period):
        groupby_rolling_max = data_t.groupby('stk', group_keys=False) \
            .rolling(window=delta_period, min_periods=1) \
            .max()
        drawdown = (data_t.values - groupby_rolling_max.values) / groupby_rolling_max

        factor = drawdown.copy()
        factor.index = data_t.index
        return factor

    def _std(self, data_t, delta_period):
        groupby_rolling_std = data_t.groupby('stk', group_keys=False) \
            .rolling(window=delta_period) \
            .std()
        factor = groupby_rolling_std.copy()
        factor.index = data_t.index
        return factor


# 9.电流因子，电势差除以电阻
class Current(Factor):
    def __init__(self, fields, prices_data, delta_period=20, voltage_zero_elevation_type='mean', resistance_type='std'):
        super().__init__(fields, prices_data)
        self.name = 'current'
        self.fields = fields
        self.prices_data = prices_data
        self.voltage_calc_price_or_returns = 'returns'
        self.resistance_calc_price_or_returns = self.voltage_calc_price_or_returns
        self.voltage_zero_elevation_type = voltage_zero_elevation_type
        self.resistance_type = resistance_type
        self.delta_period = delta_period
        self.calc_params = {'resistance_type': self.resistance_type,
                            'resistance_calc_price_or_returns': self.resistance_calc_price_or_returns,
                            'voltage_calc_price_or_returns': self.voltage_calc_price_or_returns,
                            'voltage_zero_elevation_type': self.voltage_zero_elevation_type,
                            'delta_period': self.delta_period}
        self.other_depend_factors = ['voltage', 'resistance']
        self._init_min_value = -np.inf
        self._init_max_value = np.inf

    def calc(self):
        price_t = self.prepare_data['close']
        voltage_obj = Voltage(self.fields, self.prices_data, self.delta_period,self.voltage_calc_price_or_returns,self.voltage_zero_elevation_type)
        voltage_calc_res = voltage_obj.calc()
        resistance_obj = Resistance(self.fields, self.prices_data, self.delta_period, self.resistance_calc_price_or_returns,self.resistance_type)
        resistance_calc_res = resistance_obj.calc()
        factor = voltage_calc_res / resistance_calc_res
        factor.index = price_t.index
        return factor


# 10.功率因子，电流乘以电压
class Power(Factor):
    def __init__(self, fields, prices_data, delta_period=20, voltage_zero_elevation_type='mean', resistance_type='std'):
        super().__init__(fields, prices_data)
        self.name = 'power'
        self.fields = fields
        self.prices_data = prices_data
        self.voltage_calc_price_or_returns = 'returns'
        self.voltage_zero_elevation_type = voltage_zero_elevation_type
        self.resistance_type = resistance_type
        self.delta_period = delta_period
        self.calc_params = {'resistance_type': self.resistance_type,
                            'voltage_zero_elevation_type': self.voltage_zero_elevation_type,
                            'delta_period': self.delta_period}
        self.other_depend_factors = ['voltage', 'current']
        self._init_min_value = -np.inf
        self._init_max_value = np.inf

    def calc(self):
        price_t = self.prepare_data['close']
        voltage_obj = Voltage(self.fields, self.prices_data, self.delta_period,self.voltage_calc_price_or_returns, self.voltage_zero_elevation_type)
        voltage_calc_res = voltage_obj.calc()
        current_obj = Current(self.fields, self.prices_data, self.delta_period, self.voltage_zero_elevation_type, self.resistance_type)
        current_calc_res = current_obj.calc()
        factor = current_calc_res * voltage_calc_res
        factor.index = price_t.index
        return factor


# 11.熵因子
class Entropy(Factor):
    def __init__(self, fields, prices_data, state_func=None,state_change_period=1, delta_period=5):
        super().__init__(fields, prices_data)
        self.name = 'entropy'
        self.fields = fields
        self.prices_data = prices_data
        self.state_change_period = state_change_period
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period, }
        self.other_depend_factors = []
        self.state_func = state_func
        self._init_min_value = -np.inf
        self._init_max_value = np.inf

    def calc(self):
        price_t = self.prepare_data['close']
        price_t_unstack = price_t.unstack(level=0).sort_index()
        price_t_unstack_return_t = price_t_unstack.pct_change(1)
        # 为了确保收益的结果不破坏nan值得存在性，用1来替代价格矩阵，
        # 再用价格矩阵和收益矩阵相乘时，nan值得存在性也会被保留
        p_is_not_nan = price_t_unstack.copy()
        p_is_not_nan[~p_is_not_nan.isna()] = 1
        price_t_unstack_return_t = price_t_unstack_return_t * p_is_not_nan

        # 计算Q
        # Q为两个状态的前后变化，这里定义为两个窗口的收益均值变化，也可以有别的定义，例如均值之差，方差之差，最大最小值之差等
        state_t = self._state_mean(price_t_unstack_return_t)
        state_t_change_period = state_t.shift(self.state_change_period)
        Q = state_t - state_t_change_period
        # 计算熵
        entropy = -(Q * np.log(1 + np.abs(Q)))
        entropy_s = entropy.stack(level=0, dropna=False).swaplevel().sort_index()
        factor = entropy_s.loc[price_t.index]
        return factor

    def _state_mean(self, df, step=1):
        if self.state_func is not None:
            return self.state_func(df)
        else:
            if step == 1:
                return df.rolling(window=self.delta_period) \
                    .mean()
            else:
                return df.rolling(window=self.delta_period, step=self.delta_period) \
                    .mean()


# 12.分形维度因子
class FractalDimension(Factor):
    def __init__(self, fields, prices_data, delta_period=5):
        super().__init__(fields, prices_data)
        self.name = 'fractal_dimension'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self._init_min_value = -np.inf
        self._init_max_value = np.inf

    def calc(self):
        price_t = self.prepare_data['close']
        price_t_unstack = price_t.unstack(level=0).sort_index()
        delta_period = self.delta_period
        price_t_unstack_max = price_t_unstack.rolling(window=delta_period, min_periods=1).max()
        price_t_unstack_min = price_t_unstack.rolling(window=delta_period, min_periods=1).min()
        # 计算因子
        log_t = np.log(price_t_unstack_max - price_t_unstack_min)
        factor = (1 + log_t) / (1 + log_t.shift(periods=1))
        factor = factor.replace([np.inf, -np.inf], np.nan)
        factor = factor.T.stack(level=0, dropna=True).sort_index()
        return factor

# 13. 平均价格因子
class MeanPrice(Factor):
    def __init__(self, fields, prices_data, delta_period=5):
        super().__init__(fields, prices_data)
        self.name = 'mean_price'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
    
    def calc(self):
        price_t = self.prepare_data['close']
        delta_period = self.delta_period
        # 为了确保五根k线的完整性，这里没有设置min_periods
        price_t_mean = price_t.groupby('stk',group_keys=False).rolling(window=delta_period).mean()
        # 因为groupby使得多了一个索引level，把它重新设置为计算前的索引
        price_t_mean.index = price_t.index
        # 为了确保量纲的一致性，除以今日收盘价
        factor = price_t_mean/price_t
        return factor


# 14. 平均收益绝对值因子
class gMeanReturn(Factor):
    def __init__(self, fields, prices_data, delta_period=5):
        super().__init__(fields, prices_data)
        self.name = 'gmean_return'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
    
    def calc(self):
        price_t = self.prepare_data['close']
        return_t = price_t.groupby('stk',group_keys=False).pct_change(1)
        delta_period = self.delta_period
        return_t_gmean = return_t.groupby('stk',group_keys=False)\
            .rolling(window=delta_period).apply(lambda df: df.abs().prod() ** (1 / len(df)))
        factor = return_t_gmean.copy()
        # 因为groupby使得多了一个索引level，把它重新设置为计算前的索引
        factor.index = return_t.index
        return factor
    
# 15. 今日价格位置因子,t时刻的价格处于时间窗口delta_t内，最大最小值的位置
class PricePosition(Factor):
    def __init__(self, fields, prices_data, delta_period=5):
        super().__init__(fields, prices_data)
        self.name = 'price_position'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
        
    def calc(self):
        price_t = self.prepare_data['close']
        price_t_unstack = price_t.unstack(level=0).sort_index()
        delta_period = self.delta_period
        price_t_unstack_max = price_t_unstack.rolling(window=delta_period).max()
        price_t_unstack_min = price_t_unstack.rolling(window=delta_period).min()
        price_t_unstack_position = price_t_unstack/(price_t_unstack_max + price_t_unstack_min)
        factor = price_t_unstack_position.T.stack(level=0, dropna=True).sort_index()
        return factor

# 16. 磁场因子（依靠电流因子）
class MagneticField(Factor): 
    def __init__(self, fields, prices_data, delta_period=20):
        super().__init__(fields, prices_data)
        self.name = 'magnetic_field'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self.other_depend_factors = ['current']
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
    
    def calc(self):
        price_t = self.prepare_data['close']
        current_obj = Current(self.fields, self.prices_data, self.delta_period)
        current_t = current_obj.calc()
        current_t_shift_1 = current_t.groupby('stk',group_keys=False).shift(1)
        factor = current_t - current_t_shift_1
        # 因为groupby使得多了一个索引level，把它重新设置为计算前的索引
        factor.index = price_t.index
        return factor
    
# 17. 电磁感应因子（依靠磁场因子）
class ElectromagneticInduction(Factor):
    def __init__(self, fields, prices_data, delta_period=20):
        super().__init__(fields, prices_data)
        self.name = 'electromagnetic_induction'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self.other_depend_factors = ['magnetic_field']
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
        
    def calc(self):
        price_t = self.prepare_data['close']
        magnetic_field_obj = MagneticField(self.fields, self.prices_data, self.delta_period)
        magnetic_field_t = magnetic_field_obj.calc()
        return_t = price_t.groupby('stk',group_keys=False).pct_change(1)
        factor = magnetic_field_t * return_t
        # 因为groupby使得多了一个索引level，把它重新设置为计算前的索引
        factor.index = price_t.index
        return factor

# 18. 波动因子（依靠收益）
class Volatility(Factor):
    def __init__(self, fields, prices_data, delta_period=1):
        super().__init__(fields, prices_data)
        self.name = 'volatility'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
        
    def calc(self):
        price_t = self.prepare_data['close']
        
        return_t = price_t.groupby('stk',group_keys=False).pct_change(1)
        delta_period = self.delta_period
        return_t_delta_period = return_t.shift(delta_period)
        factor = return_t/return_t_delta_period
        # 因为groupby使得多了一个索引level，把它重新设置为计算前的索引
        factor.index = price_t.index
        return factor
    
# 19. 振动周期因子（依靠波动因子）
class VibrationPeriod(Factor):
    def __init__(self, fields, prices_data, delta_period=1):
        super().__init__(fields, prices_data)
        self.name = 'vibration_period'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self.other_depend_factors = ['volatility']
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
        
    def calc(self):
        volatility_obj = Volatility(self.fields, self.prices_data, self.delta_period)
        volatility_t = volatility_obj.calc()
        factor = 1/volatility_t 
        return factor

# 20. 混沌因子
class Chaos(Factor):
    def __init__(self, fields, prices_data, delta_period=20):
        super().__init__(fields, prices_data)
        self.name = 'chaos'
        self.fields = fields
        self.prices_data = prices_data
        self.delta_period = delta_period
        self.calc_params = {'delta_period': self.delta_period}
        self._init_min_value = -np.inf
        self._init_max_value = np.inf
        
    def calc(self):
        price_t = self.prepare_data['close']
        delta_period = self.delta_period
        price_t_delta_period = price_t.groupby('stk').shift(delta_period)
        tmp_df = price_t/(price_t/price_t_delta_period - 1)
        #计算tmp的sin值
        factor = tmp_df.apply(lambda x: np.sin(x))
        factor.index = price_t.index
        return factor

if __name__ == '__main__':
    pass
