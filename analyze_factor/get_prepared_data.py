# -*- coding: utf-8 -*-
import os

import pandas as pd
import performance as pef
import prepare


def get_prepare_date_bar(date_bar_path, save_folder_path):
    save_data_path = os.path.join(save_folder_path, 'prepared_date_bar.pkl')
    # 判断save_data_path是否存在，不存在则创建
    if not os.path.exists(save_data_path):
        prepare_date_bar = prepare.PrepareDateBar(date_bar_path)
        # 设置参数
        params_dict = {
            'del_new_stks': True,  # 是否删除新股
            'start_date': None,
            'end_date': None,
            'del_paused': False,
            'fields': None
        }

        prepared_date_bar = prepare_date_bar.get_prepare_date_bar(**params_dict)
        prepared_date_bar.to_pickle(save_data_path)
    else:
        print(f"Don't need to prepare date_bar, prepared_date_bar.pkl exists,please check{save_data_path}")


def _factor_data(factor_data_path, factor_name, save_folder_path):
    factor_store = pd.HDFStore(factor_data_path, 'r')


# def get_forward_returns_total_loss_percent(forward_returns):
#     if forward_returns.index.names == ['stk','date']:
#         pass
#     elif forward_returns.index.names == ['date','stk']:
#         pass

def get_prepare_forward_returns(prepared_date_bar_path, fields, save_folder_path,
                                periods=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377)):
    save_data_path = os.path.join(save_folder_path, 'prepared_forward_returns.pkl')
    if not os.path.exists(save_data_path):
        date_bar = pd.read_pickle(prepared_date_bar_path)
        date_bar = date_bar[date_bar['paused'] != 1]
        price_df = date_bar.loc[:, fields]
        forward_returns = pef.compute_forward_returns(price_df, periods=periods)
        # forward_returns_total_loss_percent = pef.get_forward_returns_total_loss_percent(forward_returns)
        forward_returns.to_pickle(save_data_path)
        print('done')
    else:
        print(f"Don't need to prepare forward_returns, prepared_forward_returns.pkl exists,please check{save_data_path}")


def get_prepare_forward_returns_unstack(prepared_date_bar_path, fields, save_folder_path, periods=(1,)):
    save_data_path = os.path.join(save_folder_path, 'prepared_forward_returns_unstack.pkl')
    if not os.path.exists(save_data_path):
        date_bar = pd.read_pickle(prepared_date_bar_path)
        date_bar = date_bar[date_bar['paused'] != 1]
        price_df = date_bar.loc[:, fields]
        forward_returns = pef.compute_forward_returns_unstack(price_df, periods=periods)
        # forward_returns_total_loss_percent = pef.get_forward_returns_total_loss_percent(forward_returns)
        forward_returns.to_pickle(save_data_path)
        print('done')
    else:
        print(f"Don't need to prepare forward_returns, prepared_forward_returns.pkl exists,please check{save_data_path}")


if __name__ == '__main__':
    save_folder_path = r'F:\factor_lab_res\prepared_data'
    # 生成准备好的日线数据
    jq_date_bar_post_path = r'D:\QUANT_GAME\python_game\factor\factor_lab\get_date_bar\date_bar_post.pkl'
    get_prepare_date_bar(jq_date_bar_post_path,save_folder_path)
    # ========================================================================
    # 用准备好的日线数据根据给出的周期计算收益
    prepared_date_bar_path = os.path.join(save_folder_path, 'prepared_date_bar.pkl')
    # get_prepare_forward_returns_unstack(prepared_date_bar_path, ['close', ], save_folder_path)
    get_prepare_forward_returns(prepared_date_bar_path, ['close', ], save_folder_path)
