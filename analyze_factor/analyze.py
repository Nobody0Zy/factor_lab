import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting as pl
import scipy.stats as stats
from performance import data_winsorize
from scipy.stats import pearsonr, spearmanr


class zyFactorAnalyzer:
    def __init__(self,factor_name:str,periods: tuple,factor_data:pd.DataFrame,forward_returns:pd.DataFrame):
        """ 
        
        """
        self.factor_name = factor_name
        self.factor_data = factor_data
        self.factor_data.columns = [self.factor_name]
        self.forward_returns = forward_returns
        self.periods = periods
        self.norm_ic = self.calc_factor_ic(method='normal')
        self.rank_ic = self.calc_factor_ic(method='rank')
        self.norm_abs_ic = self.calc_factor_ic(method='normal',abs_tf=True)
        self.rank_abs_ic = self.calc_factor_ic(method='rank',abs_tf=True)
        
    @property
    def _column_periods(self):
        columns_periods = [f'period_{p}' for p in self.periods]
        return columns_periods
    
    def get_concat_data(self):
        factor_copy = self.factor_data.copy()
        returns_copy = self.forward_returns.copy()[self._column_periods]
        returns_copy['factor'] = factor_copy
        # 这里没有dropna，保留了包含nan的数据，因为是用pandas的corrwith计算，pandas会自动忽略nan；
        # 如果dropna了，会使得整体的数据量减少，导致计算结果不准确
        return returns_copy
    
    def plot_factor_hist(self,del_inf=False,del_qrange=None,del_range=None):
        factor_data_copy = self.factor_data.copy()
        title = f'{self.factor_name} distribution'
        if del_inf is True:
            factor_data_copy = factor_data_copy.replace([np.inf, -np.inf], np.nan)
            title = title + ' (del_inf)'
        if del_qrange is not None:
            factor_data_copy = data_winsorize(factor_data_copy,qrange=del_qrange,inclusive=True)
            title = title + f' (del_qrange: {del_qrange})'
        if del_range is not None:
            factor_data_copy = data_winsorize(factor_data_copy,range=del_range,inclusive=True)
            title = title + f' (del_range: ({del_range[0]},{del_range[1]}))'
        
        factor_data_copy.columns = [title]
        try:
            factor_data_copy.dropna().hist(bins=500)
        except Exception as e:
            print(f"an error occurred when plotting {title}, error message: {e}")
    
    def analyze_ic_res(self):
        analyze_res_df = pd.DataFrame(index=['norm_ic_mean','norm_ic_std',
                                             'norm_ic_0.02_percent','norm_ic_t',
                                             'norm_ir','norm_abs_ic',
                                             'rank_ic_mean','rank_ic_std',
                                             'rank_ic_0.02_percent','rank_ic_t',
                                             'rank_ir','rank_abs_ic',
                                             ],
                                      columns = self._column_periods)
        # 填入结果
        # norm_ic
        analyze_res_df.loc['norm_ic_mean'] = self.norm_ic.mean()
        analyze_res_df.loc['norm_ic_std'] = self.norm_ic.std()
        analyze_res_df.loc['norm_ic_0.02_percent'] = self.norm_ic[self.norm_ic.abs()>0.02].count()/self.norm_ic.count()
        # 对norm_ic进行t检验,暂定
        analyze_res_df.loc['norm_ir'] = self.norm_ic.mean()/self.norm_ic.std()
        analyze_res_df.loc['norm_abs_ic'] = self.norm_abs_ic
        # rank_ic
        analyze_res_df.loc['rank_ic_mean'] = self.rank_ic.mean()
        analyze_res_df.loc['rank_ic_std'] = self.rank_ic.std()
        analyze_res_df.loc['rank_ic_0.02_percent'] = self.rank_ic[self.rank_ic.abs()>0.02].count()/self.rank_ic.count()
        # 对rank_ic进行t检验,暂定
        analyze_res_df.loc['rank_ir'] = self.rank_ic.mean()/self.rank_ic.std()
        analyze_res_df.loc['rank_abs_ic'] = self.rank_abs_ic
        return analyze_res_df
    
    def calc_factor_ic(self,method,abs_tf=False):
        if method is None:
            method = 'rank'
        if method not in ('rank', 'normal'):
            raise ValueError("`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = 'spearman'
        elif method == 'normal':
            method = 'pearson'
    
        concat_data = self.get_concat_data()
        
        # def get_forward_returns_columns(columns):
        #     syntax = re.compile("^period_\\d+$")
        #     return columns[columns.astype('str').str.contains(syntax, regex=True)]

        
        # def src_ic(group):
        #     f = group['factor']
        #     _ic = group[get_forward_returns_columns(group.columns)] \
        #     .apply(lambda x: stats.pearsonr(x, f)[0])
        #     return _ic
        
        # print(concat_data.describe())
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     ic_tmp = concat_data.groupby('date',group_keys=False).apply(src_ic)
        # return ic_tmp
        # 这里的np.errstate是为了防止出现除0的情况, 忽略警告信息
        if abs_tf is False:
            with np.errstate(divide='ignore', invalid='ignore'):
                # concat_data = concat_data.dropna()
                ic = concat_data.groupby('date',group_keys=False)\
                    .apply(lambda df:df.drop('factor',axis=1).corrwith(df['factor'],axis=0,method=method))
            return ic
        if abs_tf is True:
            # print(concat_data)
            with np.errstate(divide='ignore', invalid='ignore'):
                abs_ic = concat_data.drop('factor',axis=1).corrwith(concat_data['factor'],axis=0,method=method)
            return abs_ic

   
    def plot_norm_ic_ts(self):
        pl.plot_ic_ts(self.norm_ic)
    
    def plot_rank_ic_ts(self):
        pl.plot_ic_ts(self.rank_ic)
    
    def plot_norm_ic_hist(self):
        pl.plot_ic_hist(self.norm_ic)
        
    def plot_norm_ic_qq(self,theoretical_dist=None):
        theoretical_dist = 'norm' if theoretical_dist is None else theoretical_dist
        theoretical_dist = getattr(stats,theoretical_dist)
        pl.plot_ic_qq(self.norm_ic,theoretical_dist=theoretical_dist)
        
    def plot_rank_ic_hist(self):
        pl.plot_ic_hist(self.rank_ic)
        
    def plot_rank_ic_qq(self,theoretical_dist=None):
        theoretical_dist = 'norm' if theoretical_dist is None else theoretical_dist
        theoretical_dist = getattr(stats,theoretical_dist)
        pl.plot_ic_qq(self.rank_ic,theoretical_dist=theoretical_dist)
    
    def plot_scatter_and_regression(self,periods_list=None,sample_num=None,rank_tf=False,title='normal'):
        factor_data = self.factor_data.copy()
        forward_returns_periods = self.forward_returns.copy()[[f'period_{p}' for p in periods_list]]
        forward_returns_periods['factor'] = factor_data
        forward_returns_periods = forward_returns_periods.replace([-np.inf,np.inf],np.nan)
        forward_returns_periods.dropna(inplace=True)       

        if sample_num is not None:
            forward_returns_periods = forward_returns_periods.sample(sample_num).dropna()

        if rank_tf is True:
            forward_returns_periods['factor'] = forward_returns_periods['factor'].rank()
            for p in periods_list:
                forward_returns_periods[f'period_{p}'] = forward_returns_periods[f'period_{p}'].rank()

            title = 'rank'
        x = forward_returns_periods['factor'].values
        xl = 'factor'
        for p in periods_list:
            y = forward_returns_periods[f'period_{p}'].values
            yl = f'period_{p}'
            title = f'factor vs forward_return_period_{p}, {title}'
            self._plot_scater_and_regression(x,y,xl,yl,title)

    def _plot_scater_and_regression(self,x:np.array,y:np.array,xl,yl,title):
        import statsmodels.api as sm
        x = sm.add_constant(x)
        model = sm.OLS(y,x)
        results = model.fit()
        # print(results.summary())
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        # print(x,y)
        # 绘制散点图
        ax.scatter(x[:,1], y, alpha=0.5)
        # 绘制拟合直线
        ax.plot(x[:,1], results.fittedvalues, 'r--')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        # 为图表添加标题
        ax.set_title(title)

    # def plot_scatter_and_mean_std(self,rolling_window=50,periods_list=None,sample_num=None,rank_tf=False,title='normal'):
    #     factor_data = self.factor_data.copy()
    #     forward_returns_periods = self.forward_returns.copy()[[f'period_{p}' for p in periods_list]]
    #     forward_returns_periods['factor'] = factor_data
    #     forward_returns_periods.dropna(inplace=True)       

    #     if sample_num is not None:
    #         forward_returns_periods = forward_returns_periods.sample(sample_num).dropna()

    #     if rank_tf is True:
    #         forward_returns_periods['factor'] = forward_returns_periods['factor'].rank()
    #         for p in periods_list:
    #             forward_returns_periods[f'period_{p}'] = forward_returns_periods[f'period_{p}'].rank()

    #         title = 'rank'
    #     x = forward_returns_periods['factor'].values
    #     xl = 'factor'
    #     for p in periods_list:
    #         y = forward_returns_periods[f'period_{p}'].values
