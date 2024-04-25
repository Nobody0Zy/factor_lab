import sys
import pickle
import pandas as pd

idx = pd.IndexSlice


class AnalyzeFactorNotes:
    def __init__(self,factor_name,start_date,end_date,med_tf=False,z_score_tf=False,demean_tf=False,sample_num=None,sample_every_date_tf=False):
        # 初始化因子处理信息
        self.factor_name = factor_name
        self.start_date = start_date
        self.end_date = end_date
        self.med_tf = med_tf
        self.z_score_tf = z_score_tf
        self.demean_tf = demean_tf
        self.sample_num = sample_num
        self.sample_every_date_tf = sample_every_date_tf
        
        # 初始化日线、收益、因子数据、因子信息路径
        self.prepared_date_bar_path = r'F:\factor_lab_res\prepared_data\prepared_date_bar.pkl'
        self.forward_returns_path = r'F:\factor_lab_res\prepared_data\prepared_forward_returns.pkl'
        self.factor_data_store_path = r'F:\factor_lab_res\prepared_data\factor_data.h5'
        self.factor_info_dict_path = r'F:\factor_lab_res\prepared_data\factor_info_dict.pkl'
        # 初始化日线、收益、因子数据
        self.prepared_date_bar = None
        self.forward_returns = None
        self.factor_data = None
    
        #初始化用于分析的数据
        self.factor_data_for_analysis = None
        self.forward_returns_for_analysis = None
    
    def load_data(self):
        # 加载数据
        self.prepared_date_bar = pd.read_pickle(self.prepared_date_bar_path)
        self.forward_returns = pd.read_pickle(self.forward_returns_path)
        factor_data_store = pd.HDFStore(self.factor_data_store_path)
        # 因子数据
        if f'/{self.factor_name}' in factor_data_store.keys():
            self.factor_data = factor_data_store[f'/{self.factor_name}']
        else:
            print('factor_data_store.keys():',factor_data_store.keys())
            raise KeyError(f'Factor data not found in the store,please check factor data of {factor_name}')
        factor_data_store.close()

    def get_factor_info(self):
        with open(self.factor_info_dict_path,'rb') as f:
            factor_info_dict = pickle.load(f)
        factor_info = factor_info_dict[self.factor_name]
        return factor_info
    
    def get_data_for_analysis(self):
        # 导入需要的包
        sys.path.append(r'D:\QUANT_GAME\python_game\factor\factor_lab\analyze_factor')
        import performance as pef
        # 获取用于分析的因子数据和收益数据
        forward_returns_for_analysis,factor_data_for_analysis = \
            pef.get_data_for_analysis(forward_returns=self.forward_returns,
                                      factor_data=self.factor_data,
                                    stk_list=None,
                                    start_date=self.start_date,end_date=self.end_date,
                                    sample_num=self.sample_num,
                                    sample_every_date_tf=self.sample_every_date_tf,
                                    med_tf=self.med_tf,z_score_tf=self.z_score_tf,demean_tf=self.demean_tf)
        self.forward_returns_for_analysis = forward_returns_for_analysis
        self.factor_data_for_analysis = factor_data_for_analysis

    def analysis_factor(self,periods=(1,2,3,5,8,13,21,34,55,89,144,233,377,)):
        # 导入diy的因子分析库
        sys.path.append(r'D:\QUANT_GAME\python_game\factor\factor_lab\analyze_factor')
        from analyze import zyFactorAnalyzer as zy_fa
        import warnings
        warnings.filterwarnings('ignore')
        if self.factor_data_for_analysis is not None and self.forward_returns_for_analysis is not None:
            zy_fa = zy_fa(factor_name=self.factor_name,periods=periods,
                        factor_data=self.factor_data_for_analysis,
                        forward_returns=self.forward_returns_for_analysis)
        else:
            raise ValueError(
                'factor_data_for_analysis or forward_returns_for_analysis is None,plese get_data_for_analysis() first'
                )
        return zy_fa
    
    