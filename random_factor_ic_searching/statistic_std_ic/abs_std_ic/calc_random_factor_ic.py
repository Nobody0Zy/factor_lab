# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

idx = pd.IndexSlice

def read_date_bar(file_path):
    return pd.read_pickle(file_path)

def get_stk_list(date_bar):
    total_stk_list = list(date_bar.index.get_level_values(0).unique())
    index_codes_list = [idx_code  for idx_code in total_stk_list if (idx_code<'sh600000')|(idx_code>'sz399000')]
    stks_list = [security  for security in total_stk_list if security not in index_codes_list]
    return stks_list

def get_prepare_data(date_bar,fields,stk_list,del_paused=False,every_date_stk_nums=None,start_date=None,end_date=None)->pd.Series:
    data = date_bar.copy()
    data.index.names = ['stk','date']
    
    if stk_list:
        data = data.loc[idx[stk_list,:],:]
        
    if every_date_stk_nums:
        if del_paused:
            data = data[data['paused']==1]
            
        data = data.groupby('date',group_keys=False).apply(lambda s:s.sample(every_date_stk_nums))
        data.sort_index(inplace=True)
        
    if start_date:
        data = data.loc[idx[:,start_date:],:]
    if end_date:
        data = data.loc[idx[:,:end_date],:]
    if start_date and end_date:
        data = data.loc[idx[:,start_date:end_date],:]
    
    data = data.loc[:,fields]
    return data

def calc_forward_returns(price_data,window_size=1)->pd.Series:
    forward_returns = price_data.groupby('stk',group_keys=False).apply(
        lambda s:s.pct_change(window_size).shift(-window_size).iloc[:-1])
    return forward_returns
    
def get_random_factor_data(forward_returns,repetitions_num=10,random_type='normal'):
    forward_returns_dropna = forward_returns.dropna()
    factor_values_num = len(forward_returns_dropna)
    if random_type == 'uniform':
        factor_values =  np.random.uniform(0,1,factor_values_num)
    elif random_type == 'normal':
        factor_values = np.random.normal(0,1,size=(factor_values_num,repetitions_num))
    factor_data = pd.DataFrame(factor_values,index=forward_returns_dropna.index,columns=['ic']*repetitions_num)
    return factor_data

def calc_total_abs_ic(factor_data,forward_returns):
    forward_returns_dropna = forward_returns.dropna()
    forward_returns.columns = ['ic']
    abs_ic = forward_returns_dropna.corrwith(factor_data['ic'],axis=0)
    return abs_ic


def calc_yearly_abs_ic(factor_data,forward_returns):
    forward_returns_dropna = forward_returns.dropna()
    forward_returns_dropna = forward_returns_dropna.copy()
    forward_returns.columns = ['ic']
    date_list = forward_returns_dropna.index.get_level_values('date').to_list()
    year_array = np.array([date//10000 for date in date_list])
    forward_returns_dropna['year'] = year_array
    factor_data['year'] = year_array
    year_abs_ic = forward_returns_dropna.groupby('year').apply(lambda s:s.corrwith(factor_data.loc[s.index,'ic'],axis=0))
    return year_abs_ic
    

def the_game_is_on():
    a_date_bar_file_path = r'D:\QUANT_GAME\python_game\pythonProject\DATA\local_stable_data\stock\CN_stock_data\dateBar.pkl'
    a_date_bar = read_date_bar(a_date_bar_file_path)
    stks_list = get_stk_list(a_date_bar)
    fields = ['close']
    prepare_price_data = get_prepare_data(a_date_bar,fields,stks_list)
    forward_returns = calc_forward_returns(prepare_price_data)
    factor_data = get_random_factor_data(forward_returns,repetitions_num=100,random_type='normal')
    abs_ic = calc_total_abs_ic(factor_data,forward_returns)
    yearly_abs_ic = calc_yearly_abs_ic(factor_data,forward_returns)
    yearly_abs_ic_df = pd.DataFrame(yearly_abs_ic,columns=['yearly_abs_ic'])
    