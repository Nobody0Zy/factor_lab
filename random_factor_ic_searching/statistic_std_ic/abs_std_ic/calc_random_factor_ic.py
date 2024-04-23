# -*- coding: utf-8 -*-

import os
import pickle
import sys

import numpy as np
import pandas as pd

idx = pd.IndexSlice


def read_date_bar(file_path):
    return pd.read_pickle(file_path)


def get_stk_list(date_bar):
    """ 
    获取股票列表，剔除指数，包含停牌股票
    """
    total_stk_list = list(date_bar.index.get_level_values(0).unique())
    index_codes_list = [idx_code for idx_code in total_stk_list if (idx_code < 'sh600000') | (idx_code > 'sz399000')]
    stks_list = [security for security in total_stk_list if security not in index_codes_list]
    return stks_list


def get_prepare_data(date_bar, fields, stk_list, del_paused=False, every_date_stk_nums=None, start_date=None,
                     end_date=None) -> pd.Series:
    """
    准备数据
    
    :param date_bar: 日线数据
    :param fields: 字段列表
    :param stk_list: 股票列表
    :param del_paused: 是否删除停牌股票
    :param every_date_stk_nums: 每日选取股票数
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 准备好的数据
    """
    data = date_bar.copy()
    data.index.names = ['stk', 'date']

    if stk_list:
        data = data.loc[idx[stk_list, :], :]

    if every_date_stk_nums:
        if del_paused:
            data = data[data['paused'] == 1]

        data = data.groupby('date', group_keys=False).apply(lambda s: s.sample(every_date_stk_nums))
        data.sort_index(inplace=True)

    if start_date:
        data = data.loc[idx[:, start_date:], :]
    if end_date:
        data = data.loc[idx[:, :end_date], :]
    if start_date and end_date:
        data = data.loc[idx[:, start_date:end_date], :]

    data = data.loc[:, fields]
    return data


def calc_forward_returns(price_data, window_size=1) -> pd.Series:
    forward_returns = price_data.groupby('stk', group_keys=False).ffill().apply(
        lambda s: s.pct_change(window_size).shift(-window_size).iloc[:-1])
    return forward_returns


def get_random_factor_data(forward_returns, repetitions_num=10, random_type='normal'):
    forward_returns_dropna = forward_returns.dropna()
    factor_values_num = len(forward_returns_dropna)
    if random_type == 'uniform':
        factor_values = np.random.uniform(0, 1, factor_values_num)
    elif random_type == 'normal':
        factor_values = np.random.normal(0, 1, size=(factor_values_num, repetitions_num))
    factor_data = pd.DataFrame(factor_values, index=forward_returns_dropna.index, columns=['ic'] * repetitions_num)
    return factor_data


def calc_total_abs_ic(factor_data, forward_returns):
    forward_returns_dropna = forward_returns.dropna()
    del forward_returns
    forward_returns_dropna.columns = ['ic']
    abs_ic = forward_returns_dropna.corrwith(factor_data['ic'], axis=0)
    return abs_ic


def calc_yearly_abs_ic(factor_data, forward_returns):
    forward_returns_dropna = forward_returns.dropna()
    forward_returns_dropna = forward_returns_dropna.copy()
    forward_returns_dropna.columns = ['ic']
    date_list = forward_returns_dropna.index.get_level_values('date')
    year_array = np.array([date // 10000 for date in date_list])
    forward_returns_dropna['year'] = year_array
    # factor_data['year'] = year_array
    year_abs_ic = forward_returns_dropna.groupby('year').apply(lambda s: s.corrwith(factor_data.loc[s.index, :], axis=0))
    year_abs_ic.dropna(axis=1, inplace=True)
    return year_abs_ic


def generate_random_factor_and_calc_total_abs_ic(forward_returns):
    generate_num = 0
    while True:
        factor_data = get_random_factor_data(forward_returns, repetitions_num=200, random_type='normal')
        abs_ic = calc_total_abs_ic(factor_data, forward_returns)
        yield list(abs_ic.values)
        generate_num += 1


def generate_random_factor_and_calc_yearly_abs_ic(forward_returns):
    generate_num = 0
    while True:
        factor_data = get_random_factor_data(forward_returns, repetitions_num=400, random_type='normal')
        yearly_abs_ic = calc_yearly_abs_ic(factor_data, forward_returns)
        yield yearly_abs_ic
        generate_num += 1


def get_abs_ic(forward_returns, iterations_num=10):
    gen = generate_random_factor_and_calc_total_abs_ic(forward_returns)
    abs_ic_list = []
    save_folder_path = r'D:\QUANT_GAME\python_game\factor\factor_lab_demo\random_factor_ic_searching\abs_ic'
    file_list = os.listdir(save_folder_path)
    file_num = len(file_list)
    for i in range(iterations_num):
        print(i)
        abs_ic_list.extend(next(gen))
        if (i + 1) % 10 == 0:
            print(f'save abs_ic_list_{file_num*10 + (i + 1)}')
            with open(os.path.join(save_folder_path, f'abs_ic_list_{file_num*10 + (i + 1)}.pkl'), 'wb') as f:
                pickle.dump(abs_ic_list, f)
            abs_ic_list = []


def get_yearly_abs_ic(forward_returns, iterations_num=10):
    gen = generate_random_factor_and_calc_yearly_abs_ic(forward_returns)
    yearly_abs_ic_dict = dict()
    save_folder_path = r'D:\QUANT_GAME\python_game\factor\factor_lab_demo\random_factor_ic_searching\yearly_abs_ic'
    file_list = os.listdir(save_folder_path)
    file_num = len(file_list)
    for i in range(iterations_num):
        print(i)
        yearly_abs_ic_df = next(gen)
        for year in yearly_abs_ic_df.index:
            if year not in yearly_abs_ic_dict.keys():
                yearly_abs_ic_dict[year] = []
            yearly_abs_ic_dict[year].extend(list(yearly_abs_ic_df.loc[year].values))
        if (i + 1) % 10 == 0:
            print(f'save yearly_abs_ic_dict_{file_num*10 + (i + 1)}')
            with open(os.path.join(save_folder_path, f'yearly_abs_ic_dict_{file_num*10 + (i + 1)}.pkl'), 'wb') as f:
                pickle.dump(yearly_abs_ic_dict, f)
            yearly_abs_ic_dict = dict()


def the_game_is_on():
    a_date_bar_file_path = r'D:\QUANT_GAME\python_game\factor\factor_lab_demo\get_date_bar\date_bar_post.pkl'
    a_date_bar = read_date_bar(a_date_bar_file_path)
    stks_list = get_stk_list(a_date_bar)
    fields = ['close']
    prepare_price_data = get_prepare_data(a_date_bar, fields, stks_list, every_date_stk_nums=1200)
    forward_returns = calc_forward_returns(prepare_price_data)
    # get_abs_ic(forward_returns, iterations_num=500)
    get_yearly_abs_ic(forward_returns, iterations_num=250)


if __name__ == '__main__':
    the_game_is_on()
