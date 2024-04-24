# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd

idx = pd.IndexSlice
class PrepareDateBar:
    """ 
    本代码用于提供因子分析的准备数据
    包括：
    读取日线数据
    获取日线数据全部股票列表
    删除新增股票数据
    返回一个需要的数据集
    """

    def __init__(self,date_bar_path):
        self.date_bar = pd.read_pickle(date_bar_path)

    @property
    def _get_stk_codes_list(self):
        """ 
        获取日线数据全部股票列表
        :param date_bar: 日线数据
        :return stks_list: 全部股票列表
        """
        total_stk_list = list(self.date_bar.index.get_level_values(0).unique())
        index_codes_list = \
            [idx_code for idx_code in total_stk_list if (idx_code<'sh600000')|(idx_code>'sz399000')]
        stk_codes_list = \
            [stk_code for stk_code in total_stk_list if stk_code not in index_codes_list]
        return stk_codes_list
    
    def _del_new_stks_data(self,data,new_stk_date_num=60):
        """ 
        删除新增股票数据,新增股票：有数据的第一天开始的60天的数据直接剔除
        """
        data = data.copy()
        del_res_data = data.groupby('stk',group_keys=False).apply(lambda df: df.iloc[new_stk_date_num:,:])
        return del_res_data
    

    
    def get_prepare_date_bar(self,del_new_stks=True,start_date=None,end_date=None,del_paused=False,fields:list=None)->pd.DataFrame:
        """ 
        获取准备的日线数据,准备的数据用于计算因子、计算收益、因子分析（计算ic）
        因此不应该在此进行抽样，抽样可以独立出来，抽样出索引再操作就行。
        :param del_new_stks: 是否删除新增股票数据
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param del_paused: 是否删除停牌数据
        :param sample_num: 样本数量
        :param every_date_tf: 是否每日抽样
        :param fields: 需要的字段列表
        :return data: 准备的日线数据
        """
        
        data = self.date_bar.copy()
        data.index.names = ['stk','date']
        stk_codes_list = self._get_stk_codes_list
        data = data.loc[idx[stk_codes_list,:],:]
        data.sort_index(inplace=True)
       
        if del_new_stks:
            data = self._del_new_stks_data(data)

        if start_date:
            data = data.loc[idx[:,start_date:],:]
        if end_date:
            data = data.loc[idx[:,:end_date],:]

        if del_paused:
            data = data.loc[data.paused==0,:]
        # 这里获取准备的日线，不应该考虑抽取股票池，这样太耦合了，不利于和其他因子分析模块的服用
        # 计算因子时，应该直接使用全部股票池，否则像随机抽样的股票，会导致因子计算结果不准确
        # if sample_num:
        #     sample_stks_pool = self.get_sample_stks_pool(data,sample_num,every_date=every_date_tf)
        #     data = data.loc[sample_stks_pool,:]
            
        if fields:
            data = data.loc[:,fields]
        
        # 剔除nan值的数据
        data = data.dropna()
        data = data.reset_index().set_index(['stk','date'])
 
        return data
    
class PrepareFactor:
    def __init__(self,factor_data_path,factor_name:str):
        self.factor_data = pd.HDFStore(factor_data_path)
        self.factor_name = factor_name
        self.factor_data_df = self.factor_data.get(self.factor_name)
        
        

class PrepareForwardReturns:
    def __init__(self,prepared_date_bar_path):
        pass
    
    def compute_forward_returns(self,data):
        pass 

class PrepareCleanData:
    def __init__(self,factor_data,forward_returns):
        pass 