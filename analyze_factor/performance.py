from datetime import datetime

import numpy as np
import pandas as pd

idx = pd.IndexSlice


def _del_returns_out_of_range(forward_returns,range,inclusive=False):
    forward_returns_tmp = forward_returns.copy()
    forward_returns_winorized = \
        data_winsorize(forward_returns_tmp, range=range, inclusive=inclusive)
    forward_returns_period_dropna = forward_returns_winorized.dropna()
    return forward_returns_period_dropna

# 剔除周期外的值,只要剔除的比例超过0.0001,就扩大区间
def del_period_returns_out_of_range(forward_returns,period,range,inclusive=False,threshold=0.0001):
    init_del_percent = 1
    h = 0.25
    while init_del_percent >= threshold:
        forward_returns_period_dropna = _del_returns_out_of_range(forward_returns,range,inclusive)
        # 检查剔除的值的比例
        del_percent = 1 - len(forward_returns_period_dropna) / len(forward_returns.dropna())
        print(f'period_{period}_del_out_of_range_{range}_percent:', del_percent)
        init_del_percent = del_percent
        h *= 1.2
        range = [range[0] - h,range[1] + h]
    return forward_returns_period_dropna

# 计算标的周期收益率
def compute_forward_returns(prices_df: pd.DataFrame, periods: tuple,range=[-0.25,0.25],inclusive=False):
    price_df = prices_df.copy()
    forward_returns = pd.DataFrame(index=price_df.index)
    for period in periods:
        print('computing forward returns for period: ', period)
        forward_returns_period = price_df.groupby('stk', group_keys=False). \
            pct_change(period).shift(-period)
        # 从价格中直接剔除掉1日收益错的值
        del_res = \
            del_period_returns_out_of_range(forward_returns_period,period,range,inclusive=inclusive)
        forward_returns[f'period_{period}'] = del_res
    return forward_returns

# 计算标的周期收益率(unstack)
def compute_forward_returns_unstack(prices_df: pd.DataFrame, periods: tuple):
    forward_returns = pd.DataFrame(index=prices_df.index)
    prices_df = prices_df['close'].copy().unstack().T
    for period in periods:
        print('computing forward returns for period: ', period)
        delta = prices_df.pct_change(period).shift(-period)
        delta_stack = delta.stack(dropna=True).swaplevel().sort_index()
        inter_idx = pd.Index.intersection(delta_stack.index, forward_returns.index)
        forward_returns[f'period_{period}'] = delta_stack.loc[inter_idx]
    return forward_returns



# 数据缩尾截尾法处理(内部)
def _data_winsorize(data, scale=None, range=None, qrange=None, inclusive=True):
    if scale is not None:
        upper = data.mean() + scale * data.std()
        lower = data.mean() - scale * data.std()
    elif range is not None:
        upper = range[1]
        lower = range[0]
    elif qrange is not None:
        upper = data.quantile(qrange[1])
        lower = data.quantile(qrange[0])
    else:
        raise ValueError('scale, range, or qrange must be provided')
    if inclusive:
        data[(data > upper)] = upper
        data[(data < lower)] = lower
    else:
        data[(data >= upper)] = np.nan
        data[(data <= lower)] = np.nan

    return data

# 数据缩尾截尾处理
def data_winsorize(data, scale=None, range=None, qrange=None, inclusive=True, inf2nan=True, axis=1):
    
    """
    data: pd.Series/pd.DataFrame/np.array, 待缩尾的序列
    scale: 标准差倍数，与 range，qrange 三选一，不可同时使用。会将位于 [mu - scale * sigma, mu + scale * sigma] 边界之外的值替换为边界值
    range: 列表， 缩尾的上下边界。与 scale，qrange 三选一，不可同时使用,[lower, upper]
    qrange: 列表，缩尾的上下分位数边界，值应在 0 到 1 之间，如 [0.05, 0.95]。与 scale，range 三选一，不可同时使用。
    inclusive: 是否将位于边界之外的值替换为边界值，默认为 True。如果为 True，则将边界之外的值替换为边界值，否则则替换为 np.nan
    inf2nan: 是否将 np.inf 和 -np.inf 替换成 np.nan，默认为 True如果为 True，在缩尾之前会先将 np.inf 和 -np.inf 替换成 np.nan，缩尾的时候不会考虑 np.nan，否则 inf 被认为是在上界之上，-inf 被认为在下界之下
    axis: 在 data 为 pd.DataFrame 时使用，沿哪个方向做标准化，默认为 1。 0 为对每列做缩尾，1 为对每行做缩尾。
    """
    if inf2nan:
        data = data.replace([np.inf, -np.inf], np.nan)
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        if axis == 0:
            data = data.apply(lambda x: _data_winsorize(x, scale, range, qrange, inclusive), axis=1)
        elif axis == 1:
            data = data.apply(lambda x: _data_winsorize(x, scale, range, qrange, inclusive), axis=0)
    elif isinstance(data, pd.Series):
        data = data.copy()
        data = _data_winsorize(data, scale, range, qrange, inclusive)
    return data
    
# z_score处理 
def _data_z_score(data: pd.DataFrame, by='date'):
    """
    param data: pd.DataFrame,index为multiindex=['stk','date'],columns为因子值
    param by: str, 按什么维度计算z-score, 默认按日期计算, 也可以按股票计算
    """
    z_score_res = data.groupby(by, group_keys=False).apply( \
        lambda x: (x - x.mean()) / x.std())
    return z_score_res

# 去均值处理
def _data_demean(data: pd.DataFrame, by='date'):
    """
    param data: pd.DataFrame,index为multiindex=['stk','date'],columns为因子值
    param by: str, 按什么维度去均值化, 默认按日期计算, 也可以按股票计算
    """
    demean_res = data.groupby(by, group_keys=False).apply( \
        lambda x: x - x.mean())

    return demean_res


def __data_mad_by_stk(stk_df,n):
    med = stk_df.median()
    mad = ((stk_df - med).abs()).median()
    # upper,lower
    upper = (med + n *1.4826* mad).values[0]
    lower = (med - n *1.4826* mad).values[0]
    # 替换掉异常值,缩尾法
    stk_df[(stk_df > upper)] = upper
    stk_df[(stk_df < lower)] = lower
    return stk_df

# 中位数去极值
def _data_mad(data: pd.DataFrame, by='date',n=3):
    """
    param data: pd.DataFrame,index为multiindex=['stk','date'],columns为因子值
    param by: str, 按什么维度计算中位数去极值, 默认按日期计算, 也可以按股票计算
    param n: int, 倍数，默认3倍
    """
    mad_res = data.groupby(by, group_keys=False).apply(lambda df:__data_mad_by_stk(df,n))
    return mad_res
    
def _winsorize_med(data: pd.DataFrame, scale=1, inclusive=True, inf2nan=True, axis=1):
    """
    data: pd.Series/pd.DataFrame/np.array, 待缩尾的序列
    scale: 倍数，默认为 1.0。会将位于 [med - scale * distance, med + scale * distance] 边界之外的值替换为边界值/np.nan
    inclusive bool 是否将位于边界之外的值替换为边界值，默认为 True。 如果为 True，则将边界之外的值替换为边界值，否则则替换为 np.nan
    inf2nan: 是否将 np.inf 和 -np.inf 替换成 np.nan，默认为 True。如果为 True，在缩尾之前会先将 np.inf 和 -np.inf 替换成 np.nan，缩尾的时候不会考虑 np.nan，否则 inf 被认为是在上界之上，-inf 被认为在下界之下
    axis: 在 data 为 pd.DataFrame 时使用，沿哪个方向做标准化，默认为 1。0 为对每列做缩尾，1 为对每行做缩尾
    """
    if inf2nan:
        data = data.replace([np.inf, -np.inf], np.nan)
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        if axis == 0:
            data = data.apply(lambda x: __winsorize_med(x, scale, inclusive), axis=1)
        elif axis == 1:
            data = data.apply(lambda x: __winsorize_med(x, scale, inclusive), axis=0)
    elif isinstance(data, pd.Series):
        data = data.copy()
        data = __winsorize_med(data, scale, inclusive)
    return data
    
def __winsorize_med(data, scale=1, inclusive=True):
    med = data.median()
    mad = ((data - med).abs()).median()
    # upper,lower
    upper = (med + scale * 1.4826 * mad).values[0]
    lower = (med - scale * 1.4826 * mad).values[0]
    # 替换掉异常值,缩尾法
    if inclusive:
        data[(data > upper)] = upper
        data[(data < lower)] = lower
    else:
        data[(data > upper)] = np.nan
        data[(data < lower)] = np.nan
    return data
    
def _get_sample_index(data, sample_num, every_date_tf):
    """
    随机抽样股票池，如果每日抽样，则每日是独立的，因此不需要设置random_state
    反之，如果不是每日抽样，则对总得股票直接抽样,
    
    :param data: 日线数据
    :param sample_num: 样本数量
    :param every_date_tf: 是否每日抽样
    :return data_sample.index
    """
    if isinstance(data,pd.Series): 
        data = data.copy()
    elif isinstance(data,pd.DataFrame):
        data = data[data.columns[0]]
    
    if every_date_tf:
        data.reset_index(inplace=True)
        data_sample = data.groupby('date', group_keys=False).apply(lambda df: df.sample(n=sample_num))
        data_sample = data_sample.set_index(['date', 'stk'])
    else:
        data_unstack = data.unstack()
        if data_unstack.index.name == 'stk':
            data_unstack_sample = data_unstack.sample(n=sample_num)  # ,random_state=42)
        elif data_unstack.index.name == 'date':
            data_unstack_sample = data_unstack.sample(n=sample_num, axis=1)  # ,random_state=42,axis=1)
            # 这里data_stack_sample 是需要dropna的，
            # 因为unstack的操作会产生原先没有的nan值，而这些nan值在data中是不存在的，
            # 有些股票还没有上市，data中没有数据，但是unstack后，填充了这些nan值
            # 再stack回去，会产生多余的nan值，所以需要dropna
            # 如果直接索引会报错
        data_stack_sample = data_unstack_sample.stack(dropna=True)  # .sort_index().reset_index()
        # data_sample = data_stack_sample.set_index(['date','stk'])
        data_sample = data_stack_sample
        data_sample.sort_index()
    return data_sample.index

# 更改index格式为datetime
def _transfrom_to_datetime_idx(data):
    if data.index.names == ['date', 'stk']:
        data = data.copy()
    elif data.index.names == ['stk', 'date']:
        data = data.copy().swaplevel()
    data_idx = data.index
    data_datetime_idx = [(datetime.strptime(str(idx[0]), '%Y%m%d'), idx[1]) for idx in data_idx]
    data.index = pd.MultiIndex.from_tuples(data_datetime_idx, names=['date', 'stk'])
    return data.sort_index()


def get_data_for_analysis(forward_returns: pd.DataFrame, factor_data: pd.Series,
                          stk_list=None, start_date=None, end_date=None,
                          sample_num=100, sample_every_date_tf=False,
                          med_tf=False, med_by='date',
                          demean_tf=False, demean_by='date',
                          z_score_tf=False, z_score_by='date',
                          ):
    # 筛选数据
    if stk_list is None:
        selected_forward_returns = forward_returns.loc[idx[:, start_date:end_date], :]
        selected_factor_data = factor_data.loc[idx[:, start_date:end_date], :]
    else:
        selected_forward_returns = forward_returns.loc[idx[stk_list, start_date:end_date], :]
        selected_factor_data = factor_data.loc[idx[stk_list, start_date:end_date]]

    # 抽样数据
    if sample_num is not None:
        # 必须从factor_data中抽样出索引
        # 因为如果从forward_returns中抽样，那么可能有些股票在factor_data中没有数据
        sample_data_idx = _get_sample_index(selected_factor_data, sample_num, sample_every_date_tf)
        intersection_idx = pd.Index.intersection(selected_forward_returns.index, sample_data_idx)
        forward_returns_sample = selected_forward_returns.loc[intersection_idx, :]
        factor_data_sample = selected_factor_data.loc[intersection_idx,:]
    else:
        intersection_idx = pd.Index.intersection(selected_forward_returns.index, selected_factor_data.index)
        forward_returns_sample = selected_forward_returns.loc[intersection_idx, :]
        factor_data_sample = selected_factor_data.loc[intersection_idx,:]
        # forward_returns_sample = selected_forward_returns.copy()
        # factor_data_sample = selected_factor_data.copy()
    # 或者使用filter, 但是filter会丢失index
    # factor_data_sample = selected_factor_data.filter(items=forward_returns_sample.index.get_level_values('stk'))
    if med_tf:
        factor_data_sample = _data_mad(factor_data_sample, med_by)

    if demean_tf:
        factor_data_sample = _data_demean(factor_data_sample, demean_by)


    if z_score_tf:
        factor_data_sample = _data_z_score(factor_data_sample, z_score_by)

        # 转换格式
    transformed_forward_returns = _transfrom_to_datetime_idx(forward_returns_sample)
    transformed_factor_data = _transfrom_to_datetime_idx(factor_data_sample)

    return transformed_forward_returns, transformed_factor_data
