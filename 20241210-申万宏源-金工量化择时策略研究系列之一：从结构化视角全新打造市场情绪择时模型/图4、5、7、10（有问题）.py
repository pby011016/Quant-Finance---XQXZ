# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:04:42 2025

@author: HUAWEI
"""

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr, pearsonr
import time
import os
import pickle
import warnings
from datetime import datetime
import random # 确保random模块被导入并使用

# 忽略警告
warnings.filterwarnings('ignore')

# 设置Tushare token
# 请注意：Tushare token是敏感信息，请确保在安全的环境下使用。
ts.set_token("4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77")
pro = ts.pro_api()

# 缓存路径
CACHE_DIR = "sws_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_data(filename, func, *args, **kwargs):
    """
    获取缓存数据，如果存在则直接加载，否则调用函数获取并缓存。
    这有助于避免重复的API请求和节省时间。
    """
    filepath = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"从缓存加载数据: {filename}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    print(f"获取新数据: {filename}")
    data = func(*args, **kwargs)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    return data

def get_sw_industry_list():
    """
    获取申万一级行业列表及其代码。
    """
    print("正在获取申万一级行业列表...")
    df = pro.index_classify(level='L1', src='SW2021')
    if df is None or df.empty:
        print("未能获取申万一级行业列表。")
        return [], []
    print(f"成功获取 {len(df)} 个申万一级行业。")
    return df['index_code'].tolist(), df['industry_name'].tolist()

def get_industry_daily(industry_codes, start_date, end_date):
    """
    获取指定行业在给定日期范围内的日行情数据。
    包括成交额、流通市值和收盘价。
    """
    all_data = []
    total_industries = len(industry_codes)
    
    for i, code in enumerate(industry_codes):
        try:
            print(f"获取行业数据 [{i+1}/{total_industries}]: {code}")
            
            # 获取行业数据
            df = pro.sw_daily(ts_code=code, start_date=start_date, end_date=end_date, 
                              fields='trade_date,amount,float_mv,close')
            if df is None or df.empty:
                print(f" - 未能获取行业 {code} 的数据，跳过。")
                continue
            df['industry_code'] = code
            all_data.append(df)
            
            # 每5个请求后随机延时3-6秒，避免API限制
            if (i + 1) % 5 == 0 and i < total_industries - 1:
                delay = random.uniform(3, 6)
                print(f"已完成 {i+1}/{total_industries}，等待 {delay:.1f} 秒...")
                time.sleep(delay)
                
        except Exception as e:
            print(f" - 获取行业 {code} 数据失败: {e}")
            # 如果失败，等待60秒后重试
            time.sleep(60)
            try:
                print(f"重试获取行业数据: {code}")
                df = pro.sw_daily(ts_code=code, start_date=start_date, end_date=end_date, 
                                  fields='trade_date,amount,float_mv,close')
                if df is None or df.empty:
                    print(f" - 重试未能获取行业 {code} 的数据，跳过。")
                    continue
                df['industry_code'] = code
                all_data.append(df)
                print(" - 重试成功")
            except Exception as e_retry:
                print(f" - 重试失败: {e_retry}，跳过行业 {code}")
    
    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()

def calculate_amt_correlation(df):
    """
    计算行业成交额占比一致性指标和行业成交额占比波动水平。
    这些指标用于衡量市场交易主线是否一致以及资金交易的频繁程度。
    """
    if df.empty:
        print("calculate_amt_correlation: 输入DataFrame为空，无法计算。")
        return pd.DataFrame(), pd.DataFrame()

    # 检查必要的列是否存在
    required_cols = ['amount', 'float_mv', 'trade_date', 'industry_code']
    if not all(col in df.columns for col in required_cols):
        print(f"calculate_amt_correlation: 缺少必要列。所需列: {required_cols}，现有列: {df.columns.tolist()}")
        return pd.DataFrame(), pd.DataFrame()

    # 计算每个行业的相对成交额（成交额/流通市值）
    # 避免除以零或NaN
    df['relative_amt'] = df['amount'] / df['float_mv'].replace(0, np.nan)
    df = df.dropna(subset=['relative_amt']) # 删除因除以零或NaN产生的NaN值
    
    if df.empty:
        print("calculate_amt_correlation: 计算relative_amt后DataFrame为空。")
        return pd.DataFrame(), pd.DataFrame()

    # 计算每日各行业的相对成交额占比
    daily_total = df.groupby('trade_date')['relative_amt'].sum().reset_index(name='total_relative_amt')
    df = pd.merge(df, daily_total, on='trade_date', how='left')
    
    # 避免除以零或NaN
    df['amt_share'] = df['relative_amt'] / df['total_relative_amt'].replace(0, np.nan)
    df = df.dropna(subset=['amt_share']) # 删除因除以零或NaN产生的NaN值

    if df.empty:
        print("calculate_amt_correlation: 计算amt_share后DataFrame为空。")
        return pd.DataFrame(), pd.DataFrame()
    
    # 按日期和行业排序，计算排名
    df['rank'] = df.groupby('trade_date')['amt_share'].rank(ascending=False, method='dense')
    
    # 按日期排序
    df = df.sort_values(['trade_date', 'industry_code'])
    
    # 计算每日与前一日排名序列的相关系数
    dates = sorted(df['trade_date'].unique())
    corr_values = []
    
    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]
        
        # 获取当前日期和前一日期的排名数据
        current_ranks = df[df['trade_date'] == current_date][['industry_code', 'rank']].set_index('industry_code')
        prev_ranks = df[df['trade_date'] == prev_date][['industry_code', 'rank']].set_index('industry_code')
        
        # 对齐行业数据
        combined = prev_ranks.join(current_ranks, how='inner', lsuffix='_prev', rsuffix='_curr')
        
        # 计算斯皮尔曼相关系数
        if len(combined) > 1:
            corr, _ = spearmanr(combined['rank_prev'], combined['rank_curr'])
            corr_values.append({'trade_date': current_date, 'amt_corr': corr})
        else:
            corr_values.append({'trade_date': current_date, 'amt_corr': np.nan})
    
    corr_df = pd.DataFrame(corr_values)
    
    # 计算20日均线和20日标准差
    corr_df['amt_corr_ma'] = corr_df['amt_corr'].rolling(20).mean()
    corr_df['amt_corr_std'] = corr_df['amt_corr'].rolling(20).std()
    corr_df['amt_corr_std_ma'] = corr_df['amt_corr_std'].rolling(20).mean()
    
    return corr_df, df

def calculate_return_amt_correlation(industry_data):
    """
    计算行业涨幅和成交额变化一致性指标。
    衡量市场资金的投资偏好或情绪是否稳定。
    """
    if industry_data.empty:
        print("calculate_return_amt_correlation: 输入DataFrame为空，无法计算。")
        return pd.DataFrame()

    # 检查必要的列是否存在
    required_cols = ['amount', 'float_mv', 'trade_date', 'industry_code', 'close']
    if not all(col in industry_data.columns for col in required_cols):
        print(f"calculate_return_amt_correlation: 缺少必要列。所需列: {required_cols}，现有列: {industry_data.columns.tolist()}")
        return pd.DataFrame()

    # 计算每个行业的相对成交额（成交额/流通市值）
    industry_data['relative_amt'] = industry_data['amount'] / industry_data['float_mv'].replace(0, np.nan)
    
    # 计算每个行业的日收益率
    industry_data = industry_data.sort_values(['industry_code', 'trade_date'])
    industry_data['return'] = industry_data.groupby('industry_code')['close'].pct_change()
    
    # 删除第一个交易日的NaN收益率和relative_amt中的NaN
    industry_data = industry_data.dropna(subset=['return', 'relative_amt'])

    if industry_data.empty:
        print("calculate_return_amt_correlation: 计算return或relative_amt后DataFrame为空。")
        return pd.DataFrame()

    # 按日期分组
    grouped = industry_data.groupby('trade_date')
    return_amt_corr = []
    
    for date, group in grouped:
        # 确保组内有足够的数据进行排名和相关性计算
        if len(group) < 2:
            return_amt_corr.append({'trade_date': date, 'return_amt_corr': np.nan})
            continue

        # 计算每个行业的收益率排名
        group['return_rank'] = group['return'].rank(ascending=False, method='dense')
        
        # 计算每个行业的相对成交额排名
        group['relative_amt_rank'] = group['relative_amt'].rank(ascending=False, method='dense')
        
        # 计算收益率排名和相对成交额排名的相关系数
        # 确保用于计算相关系数的序列长度大于1
        if len(group['return_rank']) > 1 and len(group['relative_amt_rank']) > 1:
            corr, _ = spearmanr(group['return_rank'], group['relative_amt_rank'])
            return_amt_corr.append({'trade_date': date, 'return_amt_corr': corr})
        else:
            return_amt_corr.append({'trade_date': date, 'return_amt_corr': np.nan})
    
    corr_df = pd.DataFrame(return_amt_corr)
    
    # 计算60日均线
    corr_df['return_amt_corr_ma60'] = corr_df['return_amt_corr'].rolling(60).mean()
    
    return corr_df

def calculate_gem_activity(start_date, end_date):
    """
    计算创业板成交活跃度指标。
    衡量资金对A股风险偏好的表征。
    """
    print("正在获取创业板和万得全A指数数据...")
    # 获取创业板指数数据
    gem = pro.index_daily(ts_code='399006.SZ', start_date=start_date, end_date=end_date,
                          fields='trade_date,amount')
    
    # 获取万得全A指数数据
    all_a = pro.index_daily(ts_code='881001.WI', start_date=start_date, end_date=end_date,
                            fields='trade_date,amount')
    
    if gem is None or gem.empty:
        print("未能获取创业板指数数据。")
        return pd.DataFrame()
    
    if all_a is None or all_a.empty:
        print("未能获取万得全A指数数据。")
        return pd.DataFrame()

    # 合并数据
    merged = pd.merge(gem, all_a, on='trade_date', suffixes=('_gem', '_all_a'), how='inner')
    
    if merged.empty:
        print("合并创业板和万得全A数据后为空。")
        return pd.DataFrame()

    # 计算创业板成交额占比
    merged['gem_activity'] = merged['amount_gem'] / merged['amount_all_a'].replace(0, np.nan)
    merged = merged.dropna(subset=['gem_activity']) # 删除因除以零或NaN产生的NaN值
    
    if merged.empty:
        print("计算gem_activity后DataFrame为空。")
        return pd.DataFrame()

    # 计算20日均线
    merged['gem_activity_ma20'] = merged['gem_activity'].rolling(20).mean()
    
    return merged[['trade_date', 'gem_activity', 'gem_activity_ma20']]

def calculate_margin_ratio(start_date, end_date):
    """
    计算融资余额占自由流通市值比。
    体现存量的、偏长线交易的融资资金占全市场资金的比重。
    """
    print("正在获取融资余额和A股自由流通市值数据...")
    # 获取融资余额数据
    margin = pro.margin(start_date=start_date, end_date=end_date, fields='trade_date,rz_balance')
    
    # 获取A股自由流通市值
    market_val = pro.daily_basic(start_date=start_date, end_date=end_date,
                                 fields='trade_date,float_mv')

    if margin is None or margin.empty:
        print("未能获取融资余额数据。")
        return pd.DataFrame()
    
    if market_val is None or market_val.empty:
        print("未能获取A股自由流通市值数据。")
        return pd.DataFrame()

    # **重要修正：在访问 'float_mv' 列之前，检查其是否存在**
    if 'float_mv' not in market_val.columns:
        print(f"错误：pro.daily_basic 返回的数据中缺少 'float_mv' 列。可用列: {market_val.columns.tolist()}")
        # 如果缺少关键列，则返回空DataFrame，避免后续错误
        return pd.DataFrame()

    # 聚合每日的总流通市值
    market_val = market_val.groupby('trade_date')['float_mv'].sum().reset_index()
    
    # 合并数据
    merged = pd.merge(margin, market_val, on='trade_date', how='inner')
    
    if merged.empty:
        print("合并融资余额和流通市值数据后为空。")
        return pd.DataFrame()

    # 计算融资余额占比
    merged['margin_ratio'] = merged['rz_balance'] / merged['float_mv'].replace(0, np.nan)
    merged = merged.dropna(subset=['margin_ratio']) # 删除因除以零或NaN产生的NaN值

    if merged.empty:
        print("计算margin_ratio后DataFrame为空。")
        return pd.DataFrame()

    # 计算60日均线
    merged['margin_ratio_ma60'] = merged['margin_ratio'].rolling(60).mean()
    
    return merged[['trade_date', 'margin_ratio', 'margin_ratio_ma60']]

def calculate_industry_rotation(industry_data):
    """
    计算行业轮涨补涨程度指标。
    衡量相邻时间截面上行业之间涨跌幅排序的相关性。
    """
    if industry_data.empty:
        print("calculate_industry_rotation: 输入DataFrame为空，无法计算。")
        return pd.DataFrame()

    # 检查必要的列是否存在
    required_cols = ['close', 'trade_date', 'industry_code']
    if not all(col in industry_data.columns for col in required_cols):
        print(f"calculate_industry_rotation: 缺少必要列。所需列: {required_cols}，现有列: {industry_data.columns.tolist()}")
        return pd.DataFrame()

    # 计算每个行业的日收益率
    industry_data = industry_data.sort_values(['industry_code', 'trade_date'])
    industry_data['return'] = industry_data.groupby('industry_code')['close'].pct_change()
    
    # 删除第一个交易日的NaN收益率
    industry_data = industry_data.dropna(subset=['return'])

    if industry_data.empty:
        print("calculate_industry_rotation: 计算return后DataFrame为空。")
        return pd.DataFrame()

    rotation_data = []
    dates = sorted(industry_data['trade_date'].unique())
    
    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]
        
        # 获取当前日期和前一日期的行业收益率
        current_returns = industry_data[industry_data['trade_date'] == current_date][['industry_code', 'return']]
        prev_returns = industry_data[industry_data['trade_date'] == prev_date][['industry_code', 'return']]
        
        # 合并数据
        merged = pd.merge(prev_returns, current_returns, on='industry_code', suffixes=('_prev', '_curr'), how='inner')
        
        # 确保有足够的数据进行排名和相关性计算
        if len(merged) < 2:
            rotation_data.append({'trade_date': current_date, 'rotation_corr': np.nan})
            continue

        # 计算收益率排名
        merged['rank_prev'] = merged['return_prev'].rank(ascending=False, method='dense')
        merged['rank_curr'] = merged['return_curr'].rank(ascending=False, method='dense')
        
        # 计算排名相关系数
        if len(merged['rank_prev']) > 1 and len(merged['rank_curr']) > 1:
            corr, _ = spearmanr(merged['rank_prev'], merged['rank_curr'])
            rotation_data.append({'trade_date': current_date, 'rotation_corr': corr})
        else:
            rotation_data.append({'trade_date': current_date, 'rotation_corr': np.nan})
    
    rotation_df = pd.DataFrame(rotation_data)
    
    # 计算20日均线
    rotation_df['rotation_corr_ma20'] = rotation_df['rotation_corr'].rolling(20).mean()
    
    return rotation_df

def calculate_hs300_rsi(start_date, end_date, window=14):
    """
    计算沪深300 RSI指标。
    通过特定时期内股价的变动情况计算市场买卖力量对比，来判断股价内部本质强弱。
    """
    print("正在获取沪深300指数数据计算RSI...")
    # 获取沪深300指数数据
    hs300 = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date,
                            fields='trade_date,close')
    
    if hs300 is None or hs300.empty:
        print("未能获取沪深300指数数据。")
        return pd.DataFrame()

    # 确保数据按日期排序
    hs300 = hs300.sort_values('trade_date')

    # 计算日收益率
    hs300['return'] = hs300['close'].pct_change()
    
    # 计算涨跌幅
    hs300['gain'] = np.where(hs300['return'] > 0, hs300['return'], 0)
    hs300['loss'] = np.where(hs300['return'] < 0, abs(hs300['return']), 0)
    
    # 计算平均增益和平均损失 (使用EMA平滑，更接近RSI标准计算)
    # 这里保持原代码的rolling mean，如果需要更精确的RSI，可以改为EMA
    hs300['avg_gain'] = hs300['gain'].rolling(window=window, min_periods=1).mean()
    hs300['avg_loss'] = hs300['loss'].rolling(window=window, min_periods=1).mean()
    
    # 计算相对强度 (RS)
    # 避免除以零
    hs300['rs'] = hs300['avg_gain'] / hs300['avg_loss'].replace(0, np.nan) # 替换0为NaN以避免除以零
    
    # 计算RSI
    hs300['rsi'] = 100 - (100 / (1 + hs300['rs']))
    
    # 计算20日均线
    hs300['rsi_ma20'] = hs300['rsi'].rolling(20).mean()
    
    return hs300[['trade_date', 'rsi', 'rsi_ma20']]

def get_hs300_data(start_date, end_date):
    """
    获取沪深300指数数据并计算其净值（以2017-01-03为基准）。
    """
    print("正在获取沪深300指数数据...")
    df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date,
                         fields='trade_date,close')
    
    if df is None or df.empty:
        print("未能获取沪深300指数数据。")
        return pd.DataFrame()

    # 确保数据按日期排序
    df = df.sort_values('trade_date')
    
    # 找到2017-01-03的数据作为基准
    start_value = 0
    if '20170103' in df['trade_date'].values:
        start_value = df[df['trade_date'] == '20170103']['close'].values[0]
    else:
        # 如果2017-01-03不在数据中，使用第一个交易日
        if not df.empty:
            start_value = df.iloc[0]['close']
            print(f"警告：20170103不在数据中，使用第一个交易日 {df.iloc[0]['trade_date']} 的收盘价 {start_value} 作为基准。")
        else:
            print("警告：沪深300数据为空，无法计算净值。")
            return pd.DataFrame()

    if start_value == 0: # 避免除以零
        print("警告：基准收盘价为0，无法计算净值。")
        return pd.DataFrame()
        
    df['nav'] = df['close'] / start_value
    return df[['trade_date', 'nav', 'close']]

def calculate_correlation(df, indicator_col, price_col, start_date, end_date):
    """
    计算指标与价格之间的皮尔逊相关系数。
    """
    if df.empty:
        print(f"calculate_correlation: 输入DataFrame为空，无法计算 {indicator_col} 与 {price_col} 的相关性。")
        return np.nan

    # 筛选日期范围
    filtered = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
    
    # 检查相关列是否存在
    if indicator_col not in filtered.columns or price_col not in filtered.columns:
        print(f"calculate_correlation: 缺少相关列。所需列: {indicator_col}, {price_col}，现有列: {filtered.columns.tolist()}")
        return np.nan

    # 删除缺失值
    filtered = filtered.dropna(subset=[indicator_col, price_col])
    
    if len(filtered) < 2:
        print(f"calculate_correlation: 过滤后数据点少于2个，无法计算 {indicator_col} 与 {price_col} 的相关性。")
        return np.nan
    
    # 计算皮尔逊相关系数
    corr, _ = pearsonr(filtered[indicator_col], filtered[price_col])
    
    return corr

def plot_chart(data, hs300, title, ylabel_left, ylabel_right, 
               indicator_col, nav_col='nav', corr_value=None,
               ylim_left=None, ylim_right=None, figsize=(14, 8),
               plot_start_date=None, plot_end_date=None): # Add new parameters
    """
    通用图表绘制函数，用于绘制指标与沪深300净值的双轴图。
    """
    # 合并数据
    merged = pd.merge(data, hs300, on='trade_date', how='inner') # 使用inner连接确保只有共同日期
    
    # 转换为日期格式
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged = merged.set_index('trade_date').dropna()
    
    if merged.empty:
        print(f"警告：合并后的数据为空，无法绘制图表: {title}")
        return

    # **新增：根据plot_start_date和plot_end_date筛选数据**
    if plot_start_date and plot_end_date:
        merged = merged[(merged.index >= pd.to_datetime(plot_start_date)) & 
                        (merged.index <= pd.to_datetime(plot_end_date))]
    
    if merged.empty:
        print(f"警告：在指定绘图日期范围 {plot_start_date} - {plot_end_date} 内合并后的数据为空，无法绘制图表: {title}")
        return

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese characters
    plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs correctly

    # Plot indicator
    ax1.plot(merged.index, merged[indicator_col], 
             label=ylabel_left, 
             color='#1f77b4', linewidth=2.5)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel(ylabel_left, color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    if ylim_left:
        ax1.set_ylim(ylim_left)
    
    # Plot HS300 NAV
    ax2 = ax1.twinx()
    ax2.plot(merged.index, merged[nav_col], 
             label=ylabel_right, 
             color='#ff7f0e', alpha=0.9, linewidth=2.5)
    ax2.set_ylabel(ylabel_right, color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    if ylim_right:
        ax2.set_ylim(ylim_right)
    
    # Set date format
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # **修正：将 AutoLocator 替换回 YearLocator 以确保兼容性**
    ax1.xaxis.set_major_locator(mdates.YearLocator()) 
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3)) # Add minor ticks for months

    # Add title
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Add data source
    plt.figtext(0.85, 0.01, '资料来源：Tushare，申万宏源研究', ha='center', fontsize=9)
    
    # Add annotation for right axis
    plt.text(0.02, 0.95, 'nav(右轴)', transform=ax2.transAxes, 
             color='#ff7f0e', fontsize=10, verticalalignment='top')
    
    # Add correlation value annotation
    if corr_value is not None and not np.isnan(corr_value):
        plt.figtext(0.15, 0.85, f"与沪深300价格指数的相关系数: {corr_value:.2f}", 
                    ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filename = title.replace(' ', '_').replace(':', '').replace('/', '_') + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {filename}")
    plt.show()
    plt.close()

# 主程序
def main():
    # 设置日期范围（与报告完全一致）
    start_date = '20100101'
    end_date = '20241115'
    report_start_date = '20170103'  # 报告图表起始日期
    report_end_date = '20241115'    # 报告图表结束日期
    
    # 1. 获取申万一级行业列表
    industry_codes, industry_names = get_sw_industry_list()
    if not industry_codes:
        print("无法获取行业列表，程序退出。")
        return
    
    # 2. 获取行业日行情数据（使用缓存）
    print("正在获取行业日行情数据...")
    industry_data = get_cached_data(
        "industry_data.pkl", 
        get_industry_daily, 
        industry_codes, start_date, end_date
    )
    if industry_data.empty:
        print("未能获取行业行情数据，程序退出。")
        return
    print(f"成功获取 {len(industry_data)} 条行业行情数据。")
    
    # 3. 获取沪深300净值数据
    print("正在获取沪深300指数数据...")
    hs300 = get_cached_data(
        "hs300_data.pkl",
        get_hs300_data,
        start_date, end_date
    )
    if hs300.empty:
        print("未能获取沪深300指数数据，程序退出。")
        return
    print(f"成功获取 {len(hs300)} 条沪深300指数数据。")
    
    # 4. 计算各指标并绘制图表
    print("正在计算各指标并绘制图表...")
    
    # 图4：行业成交额占比一致性与沪深300净值
    print("计算图4指标: 行业成交额占比一致性...")
    corr_df, _ = get_cached_data(
        "amt_correlation.pkl",
        calculate_amt_correlation,
        industry_data
    )
    if not corr_df.empty:
        merged_fig4 = pd.merge(corr_df, hs300, on='trade_date', how='left')
        corr_value_fig4 = calculate_correlation(merged_fig4, 'amt_corr_ma', 'close', report_start_date, report_end_date)
        plot_chart(
            corr_df, hs300, 
            title='行业成交额占比一致性与沪深300净值', 
            ylabel_left='行业成交额占比一致性(MA20)', 
            ylabel_right='沪深300净值',
            indicator_col='amt_corr_ma',
            corr_value=corr_value_fig4,
            ylim_left=(0.8, 1.0),
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date, # Add this
            plot_end_date=report_end_date      # Add this
        )
    else:
        print("无法计算图4指标，跳过图表生成。")
        
    # 图5：行业成交额占比波动水平
    print("计算图5指标: 行业成交额占比波动水平...")
    if not corr_df.empty: # 使用之前计算的corr_df
        merged_fig5 = pd.merge(corr_df, hs300, on='trade_date', how='left')
        corr_value_fig5 = calculate_correlation(merged_fig5, 'amt_corr_std_ma', 'close', report_start_date, report_end_date)
        plot_chart(
            corr_df, hs300, 
            title='行业成交额占比波动水平', 
            ylabel_left='行业成交额占比波动水平(MA20)', 
            ylabel_right='沪深300净值',
            indicator_col='amt_corr_std_ma',
            corr_value=corr_value_fig5,
            ylim_left=(0.00, 0.10),
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date, # Add this
            plot_end_date=report_end_date      # Add this
        )
    else:
        print("无法计算图5指标，跳过图表生成。")
        
    # 图7：行业涨幅和成交额变化一致性与沪深300净值
    print("计算图7指标: 行业涨幅和成交额变化一致性...")
    return_amt_corr_df = get_cached_data(
        "return_amt_correlation.pkl",
        calculate_return_amt_correlation,
        industry_data
    )
    if not return_amt_corr_df.empty:
        merged_fig7 = pd.merge(return_amt_corr_df, hs300, on='trade_date', how='left')
        corr_value_fig7 = calculate_correlation(merged_fig7, 'return_amt_corr_ma60', 'close', report_start_date, report_end_date)
        plot_chart(
            return_amt_corr_df, hs300, 
            title='行业涨幅和成交额变化一致性与沪深300净值', 
            ylabel_left='行业涨幅和成交额变化一致性(MA60)', 
            ylabel_right='沪深300净值',
            indicator_col='return_amt_corr_ma60',
            corr_value=corr_value_fig7,
            ylim_left=(-0.1, 0.3), # Adjusted
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date, # Add this
            plot_end_date=report_end_date      # Add this
        )
    else:
        print("无法计算图7指标，跳过图表生成。")
        
    # 图8：创业板成交活跃度与沪深300净值
    print("计算图8指标: 创业板成交活跃度...")
    gem_activity_df = get_cached_data(
        "gem_activity.pkl",
        calculate_gem_activity,
        start_date, end_date
    )
    if not gem_activity_df.empty:
        merged_fig8 = pd.merge(gem_activity_df, hs300, on='trade_date', how='left')
        corr_value_fig8 = calculate_correlation(merged_fig8, 'gem_activity_ma20', 'close', report_start_date, report_end_date)
        plot_chart(
            gem_activity_df, hs300, 
            title='创业板成交活跃度与沪深300净值', 
            ylabel_left='创业板成交活跃度(MA20)', 
            ylabel_right='沪深300净值',
            indicator_col='gem_activity_ma20',
            corr_value=corr_value_fig8,
            ylim_left=(0.10, 0.35),
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date, # Add this
            plot_end_date=report_end_date      # Add this
        )
    else:
        print("无法计算图8指标，跳过图表生成。")
        
    # 图9：融资余额占自由流通市值比
    print("计算图9指标: 融资余额占自由流通市值比...")
    margin_ratio_df = get_cached_data(
        "margin_ratio.pkl",
        calculate_margin_ratio,
        start_date, end_date
    )
    if not margin_ratio_df.empty:
        merged_fig9 = pd.merge(margin_ratio_df, hs300, on='trade_date', how='left')
        corr_value_fig9 = calculate_correlation(merged_fig9, 'margin_ratio_ma60', 'close', report_start_date, report_end_date)
        plot_chart(
            margin_ratio_df, hs300, 
            title='融资余额占自由流通市值比', 
            ylabel_left='融资余额占比(MA60)', 
            ylabel_right='沪深300净值',
            indicator_col='margin_ratio_ma60',
            corr_value=corr_value_fig9,
            ylim_left=(0.036, 0.050), # Adjusted
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date, # Add this
            plot_end_date=report_end_date      # Add this
        )
    else:
        print("无法计算图9指标，跳过图表生成。")
        
    # 图10：沪深300 RSI与指数净值变化趋势
    print("计算图10指标: 沪深300 RSI...")
    hs300_rsi_df = get_cached_data(
        "hs300_rsi.pkl",
        calculate_hs300_rsi,
        start_date, end_date
    )
    if not hs300_rsi_df.empty:
        merged_fig10 = pd.merge(hs300_rsi_df, hs300, on='trade_date', how='left')
        corr_value_fig10 = calculate_correlation(merged_fig10, 'rsi_ma20', 'close', report_start_date, report_end_date)
        plot_chart(
            hs300_rsi_df, hs300, 
            title='沪深300 RSI与指数净值变化趋势', 
            ylabel_left='沪深300 RSI(MA20)', 
            ylabel_right='沪深300净值',
            indicator_col='rsi_ma20',
            corr_value=corr_value_fig10,
            ylim_left=(10, 80), # Adjusted
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date, # Add this
            plot_end_date=report_end_date      # Add this
        )
    else:
        print("无法计算图10指标，跳过图表生成。")
        
    # 图11：行业轮涨补涨程度与指数净值变化趋势
    print("计算图11指标: 行业轮涨补涨程度...")
    rotation_df = get_cached_data(
        "industry_rotation.pkl",
        calculate_industry_rotation,
        industry_data
    )
    if not rotation_df.empty:
        merged_fig11 = pd.merge(rotation_df, hs300, on='trade_date', how='left')
        corr_value_fig11 = calculate_correlation(merged_fig11, 'rotation_corr_ma20', 'close', report_start_date, report_end_date)
        plot_chart(
            rotation_df, hs300, 
            title='行业轮涨补涨程度与指数净值变化趋势', 
            ylabel_left='行业轮涨补涨程度(MA20)', 
            ylabel_right='沪深300净值',
            indicator_col='rotation_corr_ma20',
            corr_value=corr_value_fig11,
            ylim_left=(0.1, 0.5), # Adjusted
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date, # Add this
            plot_end_date=report_end_date      # Add this
        )
    else:
        print("无法计算图11指标，跳过图表生成。")
        
    print("所有图表生成完成！")

if __name__ == '__main__':
    main()
