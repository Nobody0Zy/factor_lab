import os
import pickle

import pandas as pd
from def_factor_repo import FactorRepo as frepo


def _get_exist_factor_name_list():
    exist_factor_name_list = []
    # 导入已经计算好的因子信息
    factor_info_dict_path = r'F:\factor_lab_res\prepared_data\factor_info_dict.pkl'
    if not os.path.exists(factor_info_dict_path):
        return exist_factor_name_list
    else:
        with open(factor_info_dict_path, 'rb') as f:
            factor_info_dict = pickle.load(f)
        factor_info_name = factor_info_dict.keys()
        exist_factor_name_list.extend(factor_info_name)
        return exist_factor_name_list


def get_need_factor_name_list(total_factor_name_list, del_exist_factor_name=True, need_factor_name_list=None,
                              del_factor_name_list=None):
    # 如果需要计算的因子名称列表为空,
    # 则根据是否需要删除已经计算好的因子名称，计算出需要计算的因子名称列表
    # 如果需要计算的因子名称列表不为空，则直接返回
    if need_factor_name_list is None:
        if del_exist_factor_name:
            exist_factor_name_list = _get_exist_factor_name_list()
            need_factor_name_list = list(set(total_factor_name_list) - set(exist_factor_name_list))
        else:
            need_factor_name_list = total_factor_name_list
    else:
        need_factor_name_list = need_factor_name_list

    # 剔除指定因子名称
    if del_factor_name_list is not None:
        need_factor_name_list = list(set(need_factor_name_list) - set(del_factor_name_list))
    return need_factor_name_list


def calc_and_save_factor_values(prepared_date_bar_file_path, factor_name_list, ):
    # 读取原先准备好的日线数据，全部样本数据，还没有对股票进行抽样
    # prepared_date_bar_file_path = r''
    prepared_date_bar = pd.read_pickle(prepared_date_bar_file_path)
    # 这里还是需要剔除停牌，停牌的时候就不计算因子，否则会因为用前值填充造成有计算结果，与实际不符和
    prepared_date_bar = prepared_date_bar[prepared_date_bar['paused'] != 1]
    # 读取因子定义的类，并计算因子值，保存到指定路径
    for factor_name in factor_name_list:
        print(f'computing factor: {factor_name}')
        if factor_name not in factor_repo_dict.keys():
            raise ValueError(f"{factor_name} is not in the factor repo.")
        else:
            factor_def = factor_repo_dict[factor_name]
            factor_obj = factor_def(['close', ], prepared_date_bar)
            factor_value = factor_obj.calc_factor_and_save_factor_info()
            save_path = factor_obj.factor_info['save_h5_path']
            factor_value_store = pd.HDFStore(save_path)
            factor_value_store[factor_name] = factor_value
            factor_value_store.close()
    print('factor computing finished.')


if __name__ == '__main__':
    """
    这里因子的计算里的参数都是按照默认值，如果需要修改，需要到因子定义里修改。
    因子定义的路径是：
    r'D:\QUANT_GAME\python_game\factor\factor_lab\def_factor\def_factor_repo.py'
    """
    factor_repo_dict = frepo().factor_repo_dict
    prepared_date_bar_file_path = r'F:\factor_lab_res\prepared_data\prepared_date_bar.pkl'
    factor_name_list = factor_repo_dict.keys()

    # need_calc_factor_name_list = get_need_factor_name_list(factor_name_list, del_exist_factor_name=False)
    need_calc_factor_name_list = ['chaos']
    print('need_calc_factor_name_list:\n', need_calc_factor_name_list)
    print('need_calc_factor_name_list num:', len(need_calc_factor_name_list))
    calc_and_save_factor_values(prepared_date_bar_file_path, need_calc_factor_name_list)
