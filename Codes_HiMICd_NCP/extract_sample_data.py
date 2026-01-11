import os
import pandas as pd
import numpy as np
from pathlib import Path


# def extract_random_samples(base_path, output_filename="Trainsample.csv"):
#     """
#     从2003-2020年的文件夹中读取Test_all.csv，根据month和day随机提取54个均匀分布的点
#
#     参数:
#     base_path: 基础路径
#     output_filename: 输出文件名
#     """
#
#     all_samples = []  # 存储所有年份的样本
#
#     for year in range(2003, 2021):  # 2003到2020年
#         # 构建文件路径
#         folder_path = Path(base_path) / f"{year}"
#         file_path = folder_path /"humidity"/ "Train_all.csv"
#
#         print(f"处理 {year} 年的数据...")
#
#         # 检查文件是否存在
#         if not file_path.exists():
#             print(f"警告: {file_path} 不存在，跳过该年份")
#             continue
#
#         try:
#             # 读取CSV文件
#             df = pd.read_csv(file_path)
#
#             # 确保数据中有month和day列
#             if 'month' not in df.columns or 'day' not in df.columns:
#                 print(f"警告: {file_path} 中缺少month或day列，跳过该年份")
#                 continue
#
#             # 创建日期标识列（用于确保均匀分布）
#             df['date_id'] = df['month'].astype(str) + '-' + df['day'].astype(str)
#
#             # 获取所有唯一的日期组合
#             unique_dates = df['date_id'].unique()
#
#             # 如果数据量不足54个日期，则使用所有日期
#             if len(unique_dates) < 54:
#                 print(f"警告: {year}年只有{len(unique_dates)}个唯一日期，使用所有日期")
#                 selected_dates = unique_dates
#             else:
#                 # 随机选择54个均匀分布的日期
#                 selected_dates = np.random.choice(unique_dates, size=54, replace=False)
#
#             # 从每个选中的日期中随机选择一个样本
#             selected_samples = []
#             for date in selected_dates:
#                 date_data = df[df['date_id'] == date]
#                 if len(date_data) > 0:
#                     random_sample = date_data.sample(n=1)
#                     selected_samples.append(random_sample)
#
#             if selected_samples:
#                 # 合并该年份的所有样本
#                 year_samples = pd.concat(selected_samples, ignore_index=True)
#                 year_samples['source_year'] = year  # 添加来源年份标识
#                 all_samples.append(year_samples)
#                 print(f"{year}年成功抽取 {len(selected_samples)} 个样本")
#             else:
#                 print(f"警告: {year}年没有成功抽取到样本")
#
#         except Exception as e:
#             print(f"处理 {year} 年数据时出错: {e}")
#             continue
#
#     if all_samples:
#         # 合并所有年份的样本
#         final_df = pd.concat(all_samples, ignore_index=True)
#
#         # 删除临时列
#         if 'date_id' in final_df.columns:
#             final_df = final_df.drop('date_id', axis=1)
#
#         # 保存结果
#         output_path = Path(base_path) / output_filename
#         final_df.to_csv(output_path, index=False)
#         print(f"\n成功合并所有样本，共 {len(final_df)} 行数据")
#         print(f"结果已保存到: {output_path}")
#
#         # 显示统计信息
#         print("\n各年份样本数量统计:")
#         year_counts = final_df['source_year'].value_counts().sort_index()
#         for year, count in year_counts.items():
#             print(f"  {year}年: {count} 个样本")
#
#     else:
#         print("没有成功抽取到任何样本")
#
#
# # 使用示例
# if __name__ == "__main__":
#     base_path = r"E:\high-resolution atmospheric moisture\Data\2 Split Data"
#
#     # 执行抽取和合并
#     extract_random_samples(base_path)


# ==============================================================================
import os
import pandas as pd
import numpy as np

# 输入与输出路径
input_dir = r"D:\新建文件夹\fsdownload\TrainModel_PredictedData"
output_dir = r"D:\pycharm_code\HiTIC-NCP-main\Data Samples_HiMICd_NCP\Fig4 data"

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹下所有csv文件
for file_name in os.listdir(input_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_name} ...")

        # 读取csv
        df = pd.read_csv(file_path)

        # 确保有year, month, day列
        if not all(col in df.columns for col in ["year", "month", "day"]):
            print(f"⚠️ 文件 {file_name} 缺少 year/month/day 列，已跳过。")
            continue

        # 创建日期列并按日期排序
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df.sort_values("date").reset_index(drop=True)

        # 均匀采样 1000 个点
        n_samples = 1000
        if len(df) > n_samples:
            idx = np.linspace(0, len(df) - 1, n_samples, dtype=int)
            sampled_df = df.iloc[idx]
        else:
            # 如果不足 1000 行，就全部保留
            sampled_df = df.copy()
            print(f"⚠️ 文件 {file_name} 只有 {len(df)} 行，未达到1000。")

        # 保存文件
        output_path = os.path.join(output_dir, f"sampled_{file_name}")
        sampled_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存采样文件：{output_path}")

print("🎉 所有文件采样完成！")





