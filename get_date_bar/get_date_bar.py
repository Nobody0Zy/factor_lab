import os

import pandas as pd
import psutil


def get_date_bar(folder_path):
    file_list = os.listdir(folder_path)
    date_bar_list = []

    for file in file_list:
        # 检测date_bar_list内存，如果大于10g，则报错
        # 内存检测
        memory_info = psutil.Process().memory_info()
        memory_usage = memory_info.rss / 1024/1024/1024
        print(f"Memory usage: {memory_usage} GB")
        if memory_usage > 10:
            raise Exception("Memory usage is too large, please check the memory usage and try again.")

        print(f"Loading {file}...")
        every_date_bar = pd.read_pickle(os.path.join(folder_path, file))
        date_bar_list.append(every_date_bar)
    date_bar_total_df = pd.concat(date_bar_list)

    return date_bar_total_df


if __name__ == '__main__':
    folder_path = r'\\10.88.3.254\01临时共享资源\程帆\jq\jqDay_post'
    try:
        date_bar = get_date_bar(folder_path)
    except MemoryError as e:
        print(e)
        exit()

    date_bar.to_pickle(r'D:\QUANT_GAME\python_game\factor\factor_lab\get_date_bar\date_bar_post.pkl')
