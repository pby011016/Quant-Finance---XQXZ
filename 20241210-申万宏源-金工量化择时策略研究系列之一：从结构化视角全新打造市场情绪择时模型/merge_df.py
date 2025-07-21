import pandas as pd

# def get_50etf_vix_data(start_date, end_date):
#     """获取50ETF波指数据"""
#     return get_cached_data(
#         f"sws_data_cache/vix_50etf.pkl",
#         get_vix_data,
#         '50ETF', start_date, end_date
#     )

# def get_300_vix_data(start_date, end_date):
#     """获取300波指数据"""
#     return get_cached_data(
#         f"vix_300_{start_date}_{end_date}.pkl",
#         get_vix_data,
#         '300', start_date, end_date
#     )

df1 = pd.read_pickle("sws_data_cache/vix_50etf_20250206_20250704.pkl")
df2 = pd.read_pickle("sws_data_cache/vix_50etf.pkl")

print("sws_data_cache/vix_50etf_20250206_20250704.pkl head:")
print(df1.head())
print("sws_data_cache/vix_50etf_20250206_20250704.pkl tail:")
print(df1.tail())

print("sws_data_cache/vix_50etf.pkl head:")
print(df2.head())
print("sws_data_cache/vix_50etf.pkl tail:")
print(df2.tail())

# 合并数据
df_combined = pd.concat([df2, df1], ignore_index=True)
# # 去重
# df_combined_filtered = df_combined.drop_duplicates(subset=['trade_date'], keep='last')

# 保存合并后的数据
df_combined.to_pickle("sws_data_cache/vix_50etf_20191223_20250626.pkl")

