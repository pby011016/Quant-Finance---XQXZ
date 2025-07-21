# -*- coding: utf-8 -*-
"""修改后的完整代码"""
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
import random
import requests # Import requests for exception handling

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置当前日期为最新
current_date = datetime.now().strftime("%Y%m%d")

# 忽略警告
warnings.filterwarnings('ignore')

# 设置Tushare token
# Increase default timeout for Tushare API calls
ts.set_token("4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77")
# Set a longer timeout for the Tushare Pro API client
pro = ts.pro_api(timeout=60) # Increased timeout to 60 seconds

# 缓存路径
CACHE_DIR = "sws_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_data(filename, func, *args, **kwargs):
    """获取或缓存数据"""
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
    """获取申万一级行业列表"""
    print("正在获取申万一级行业列表...")
    df = pro.index_classify(level='L1', src='SW2021')
    if df is None or df.empty:
        print("未能获取申万一级行业列表。")
        return [], []
    print(f"成功获取 {len(df)} 个申万一级行业。")
    return df['index_code'].tolist(), df['industry_name'].tolist()

def get_industry_daily(industry_codes, start_date, end_date):
    """获取行业日行情数据"""
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
            
            # 每5个请求后随机延时3-6秒
            if (i + 1) % 5 == 0 and i < total_industries - 1:
                delay = random.uniform(3, 6)
                print(f"已完成 {i+1}/{total_industries}，等待 {delay:.1f} 秒...")
                time.sleep(delay)
                
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            print(f" - 获取行业 {code} 数据超时或连接错误: {e}")
            print(" - 尝试重试或检查网络/代理设置...")
            time.sleep(60) # Wait longer on network errors
        except Exception as e:
            print(f" - 获取行业 {code} 数据失败: {e}")
            time.sleep(30) # Wait longer on other errors
    
    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()

def calculate_amt_correlation(df):
    """计算行业成交额占比一致性指标"""
    # 计算相对成交额
    df['relative_amt'] = df['amount'] / df['float_mv']
    
    # 计算每日各行业的相对成交额占比
    daily_total = df.groupby('trade_date')['relative_amt'].sum().reset_index(name='total_relative_amt')
    df = pd.merge(df, daily_total, on='trade_date')
    df['amt_share'] = df['relative_amt'] / df['total_relative_amt']
    
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

def calculate_industry_concentration(industry_data):
    """计算行业集中度指标"""
    if industry_data.empty:
        return pd.DataFrame()

    # 计算每日行业集中度
    concentration_data = []
    
    for trade_date, group in industry_data.groupby('trade_date'):
        if len(group) < 5:
            concentration_data.append({'trade_date': trade_date, 'concentration': np.nan})
            continue
        
        # Ensure 'relative_amt' is available and not all NaNs in the group
        if group['relative_amt'].isnull().all():
            concentration_data.append({'trade_date': trade_date, 'concentration': np.nan})
            continue

        top5 = group.nlargest(5, 'relative_amt')
        top5_avg = top5['relative_amt'].mean()
        
        # 计算全市场平均换手率
        market_avg = group['relative_amt'].mean()
        
        # 计算集中度
        if market_avg == 0 or pd.isna(market_avg): # Handle potential NaN or zero
            concentration_data.append({'trade_date': trade_date, 'concentration': np.nan})
        else:
            concentration = top5_avg / market_avg
            concentration_data.append({'trade_date': trade_date, 'concentration': concentration})
    
    conc_df = pd.DataFrame(concentration_data)
    
    # 计算20日均线
    conc_df['concentration_ma20'] = conc_df['concentration'].rolling(20).mean()
    
    return conc_df

def calculate_return_amt_correlation(industry_data):
    """计算行业涨幅和成交额变化一致性指标"""
    if industry_data.empty:
        return pd.DataFrame()

    # 计算相对成交额
    industry_data['relative_amt'] = industry_data['amount'] / industry_data['float_mv'].replace(0, np.nan)
    
    # 计算日收益率
    industry_data = industry_data.sort_values(['industry_code', 'trade_date'])
    industry_data['return'] = industry_data.groupby('industry_code')['close'].pct_change()
    industry_data = industry_data.dropna(subset=['return', 'relative_amt'])

    # 按日期分组计算相关性
    return_amt_corr = []
    
    for date, group in industry_data.groupby('trade_date'):
        if len(group) < 2:
            return_amt_corr.append({'trade_date': date, 'return_amt_corr': np.nan})
            continue

        # Check if there's enough variation for ranking
        # Added check for unique values to prevent spearmanr errors on constant data
        if group['return'].nunique() < 2 or group['relative_amt'].nunique() < 2:
            return_amt_corr.append({'trade_date': date, 'return_amt_corr': np.nan})
            continue

        # 计算排名
        group['return_rank'] = group['return'].rank(ascending=False, method='dense')
        group['relative_amt_rank'] = group['relative_amt'].rank(ascending=False, method='dense')
        
        # 计算相关系数
        if len(group['return_rank']) > 1:
            corr, _ = spearmanr(group['return_rank'], group['relative_amt_rank'])
            return_amt_corr.append({'trade_date': date, 'return_amt_corr': corr})
        else:
            return_amt_corr.append({'trade_date': date, 'return_amt_corr': np.nan})
    
    corr_df = pd.DataFrame(return_amt_corr)
    
    # 计算60日均线
    corr_df['return_amt_corr_ma60'] = corr_df['return_amt_corr'].rolling(60).mean()
    
    return corr_df

def calculate_gem_activity(start_date, end_date):
    """计算创业板成交活跃度（增强版）"""
    print("正在获取创业板和全市场成交额数据...")
    
    try:
        # 获取创业板指数成交额 (399006.SZ)
        gem_daily = pro.index_daily(ts_code='399006.SZ', start_date=start_date, end_date=end_date,
                                    fields='trade_date,amount')
        if gem_daily is None or gem_daily.empty:
            print("未能获取创业板指数 (399006.SZ) 成交额数据。")
            return pd.DataFrame()
        
        gem_daily.rename(columns={'amount': 'amount_gem'}, inplace=True)

        # 获取全市场A股成交额 (使用 pro.daily_info 的 amount 字段作为全市场成交额的近似)
        print("正在获取全市场成交额数据 (使用 pro.daily_info)...")
        all_a_daily = pro.daily_info(start_date=start_date, end_date=end_date, fields='trade_date,amount')
        
        if all_a_daily is None or all_a_daily.empty:
            print("无法获取全市场成交额数据")
            return pd.DataFrame()
        all_a_daily.rename(columns={'amount': 'amount_all_a'}, inplace=True)
        
        # 统一日期格式
        gem_daily['trade_date'] = gem_daily['trade_date'].astype(str)
        all_a_daily['trade_date'] = all_a_daily['trade_date'].astype(str)
        
        # 合并数据
        # Use outer merge and then drop NaNs to ensure maximum date coverage
        merged = pd.merge(gem_daily, all_a_daily, on='trade_date', how='outer')
        
        # Fill NaNs if one series has more dates than the other
        merged['amount_gem'] = merged['amount_gem'].fillna(method='ffill').fillna(method='bfill')
        merged['amount_all_a'] = merged['amount_all_a'].fillna(method='ffill').fillna(method='bfill')

        # Drop rows where amount_all_a is 0 or NaN to avoid division by zero
        merged = merged[merged['amount_all_a'].replace(0, np.nan).notna()]
        
        # Calculate gem activity
        merged['gem_activity'] = merged['amount_gem'] / merged['amount_all_a']
        merged = merged.dropna(subset=['gem_activity']) # Drop if activity is NaN

        # Calculate 20-day moving average
        merged['gem_activity_ma20'] = merged['gem_activity'].rolling(20, min_periods=1).mean()
        
        return merged[['trade_date', 'gem_activity', 'gem_activity_ma20']]
    
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        print(f"计算创业板活跃度时发生超时或连接错误: {e}")
        print("请检查您的网络连接或Tushare代理设置。")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    except Exception as e:
        print(f"计算创业板活跃度时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_margin_ratio(start_date, end_date):
    """计算融资余额占比"""
    print("正在获取融资余额和A股流通市值数据...")
    
    # 获取融资余额 (Tushare rzye 字段单位为 元)
    margin = pro.margin(start_date=start_date, end_date=end_date, fields='trade_date,rzye')
    if margin is None or margin.empty:
        print("未能获取融资余额数据。")
        return pd.DataFrame()
    
    # 获取A股流通市值 (Tushare circ_mv 字段单位为 万元)
    # 汇总每日所有股票的流通市值
    market_val_raw = pro.daily_basic(start_date=start_date, end_date=end_date, fields='trade_date,circ_mv')
    if market_val_raw is None or market_val_raw.empty:
        print("未能获取A股流通市值数据。")
        return pd.DataFrame()

    market_val = market_val_raw.groupby('trade_date')['circ_mv'].sum().reset_index()
    market_val.rename(columns={'circ_mv': 'total_circ_mv'}, inplace=True) # Rename for clarity

    # 合并数据
    merged = pd.merge(margin, market_val, on='trade_date', how='inner')
    
    # 计算融资余额占比
    # Tushare rzye (融资余额) 单位是 元
    # Tushare circ_mv (流通市值) 单位是 万元
    # 报告图9的Y轴范围是 0.036 到 0.050，这通常表示的是百分比的数值形式 (即 3.6% 到 5.0%)。
    # 为了匹配这个范围，我们需要将融资余额（元）除以流通市值（元）。
    # total_circ_mv (万元) * 10000 (元/万元) = total_circ_mv (元)
    # 所以： margin_ratio = rzye (元) / (total_circ_mv (万元) * 10000)
    merged['margin_ratio'] = merged['rzye'] / (merged['total_circ_mv'] * 10000).replace(0, np.nan)
    merged = merged.dropna(subset=['margin_ratio'])
    
    # 计算60日均线
    merged['margin_ratio_ma60'] = merged['margin_ratio'].rolling(60).mean()
    
    return merged[['trade_date', 'margin_ratio', 'margin_ratio_ma60']]

def calculate_industry_rotation(industry_data):
    """计算行业轮涨补涨程度指标"""
    if industry_data.empty:
        return pd.DataFrame()

    # 计算日收益率
    industry_data = industry_data.sort_values(['industry_code', 'trade_date'])
    industry_data['return'] = industry_data.groupby('industry_code')['close'].pct_change()
    industry_data = industry_data.dropna(subset=['return'])

    # 计算轮动相关性
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
        
        if len(merged) < 2:
            rotation_data.append({'trade_date': current_date, 'rotation_corr': np.nan})
            continue

        # Check if there's enough variation for ranking
        # If all values are the same, spearmanr will return NaN or error. Ensure at least 2 unique values.
        if merged['return_prev'].nunique() < 2 or merged['return_curr'].nunique() < 2:
            rotation_data.append({'trade_date': current_date, 'rotation_corr': np.nan})
            continue

        # 计算收益率排名
        merged['rank_prev'] = merged['return_prev'].rank(ascending=False, method='dense')
        merged['rank_curr'] = merged['return_curr'].rank(ascending=False, method='dense')
        
        # 计算排名相关系数
        if len(merged['rank_prev']) > 1:
            corr, _ = spearmanr(merged['rank_prev'], merged['rank_curr'])
            rotation_data.append({'trade_date': current_date, 'rotation_corr': corr})
        else:
            rotation_data.append({'trade_date': current_date, 'rotation_corr': np.nan})
    
    rotation_df = pd.DataFrame(rotation_data)
    
    # 计算20日均线
    rotation_df['rotation_corr_ma20'] = rotation_df['rotation_corr'].rolling(20).mean()
    
    return rotation_df

def calculate_hs300_rsi(start_date, end_date, window=14):
    """计算沪深300 RSI指标"""
    print("正在获取沪深300指数数据计算RSI...")
    # 获取沪深300指数数据
    hs300 = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date,
                            fields='trade_date,close')
    
    if hs300 is None or hs300.empty:
        return pd.DataFrame()

    # 确保数据按日期排序
    hs300 = hs300.sort_values('trade_date')

    # 计算日收益率
    hs300['return'] = hs300['close'].pct_change()
    
    # 计算涨跌幅
    hs300['gain'] = np.where(hs300['return'] > 0, hs300['return'], 0)
    hs300['loss'] = np.where(hs300['return'] < 0, abs(hs300['return']), 0)
    
    # Calculate average gain and average loss using Wilder's smoothing method (Exponential Moving Average)
    hs300['avg_gain'] = hs300['gain'].ewm(com=window-1, adjust=False, min_periods=window).mean()
    hs300['avg_loss'] = hs300['loss'].ewm(com=window-1, adjust=False, min_periods=window).mean()
    
    # Calculate Relative Strength (RS)
    # Handle division by zero for avg_loss
    hs300['rs'] = hs300['avg_gain'] / hs300['avg_loss'].replace(0, np.nan)
    
    # Calculate RSI
    hs300['rsi'] = 100 - (100 / (1 + hs300['rs']))
    
    # Calculate 20-day moving average
    hs300['rsi_ma20'] = hs300['rsi'].rolling(20).mean()
    
    return hs300[['trade_date', 'rsi', 'rsi_ma20']]

def get_hs300_data(start_date, end_date):
    """获取沪深300指数数据"""
    print("正在获取沪深300指数数据...")
    df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date,
                            fields='trade_date,close')
    
    if df is None or df.empty:
        print("未能获取沪深300指数数据。")
        return pd.DataFrame()

    # 确保数据按日期排序
    df = df.sort_values('trade_date')
    
    # 找到基准日期的数据
    # Use .iloc[0]['close'] if '20170103' is not in the dataframe
    start_value = df[df['trade_date'] == '20170103']['close'].values[0] if '20170103' in df['trade_date'].values else df.iloc[0]['close']
        
    df['nav'] = df['close'] / start_value
    return df[['trade_date', 'nav', 'close']]

def calculate_correlation(df, indicator_col, price_col, start_date, end_date):
    """计算指标与价格的相关性"""
    if df.empty:
        return np.nan

    # 筛选日期范围
    filtered = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
    
    # 检查相关列是否存在
    if indicator_col not in filtered.columns or price_col not in filtered.columns:
        print(f"警告: 关联计算中缺少列 '{indicator_col}' 或 '{price_col}'")
        return np.nan

    # 删除缺失值
    filtered = filtered.dropna(subset=[indicator_col, price_col])
    
    if len(filtered) < 2:
        print(f"警告: 在指定日期范围内用于关联计算的数据点少于2个，无法计算关联性。")
        return np.nan
    
    # 计算皮尔逊相关系数
    corr, _ = pearsonr(filtered[indicator_col], filtered[price_col])
    
    return corr

def plot_chart(data, hs300, title, ylabel_left, ylabel_right,
               indicator_col, nav_col='nav', corr_value=None,
               ylim_left=None, ylim_right=None, figsize=(14, 8),
               plot_start_date=None, plot_end_date=None):
    """绘制指标与沪深300净值的双轴图"""
    # Ensure date formats are consistent strings before merging
    data_copy = data.copy()
    hs300_copy = hs300.copy()
    data_copy['trade_date'] = data_copy['trade_date'].astype(str)
    hs300_copy['trade_date'] = hs300_copy['trade_date'].astype(str)

    # Use outer merge to keep all dates
    merged = pd.merge(data_copy, hs300_copy, on='trade_date', how='outer')

    # Convert to datetime and set index
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged = merged.sort_values('trade_date').set_index('trade_date')

    # Fill missing values for the columns to be plotted
    # Use ffill to carry forward last valid observation
    if indicator_col in merged.columns:
        merged[indicator_col] = merged[indicator_col].fillna(method='ffill')
    if nav_col in merged.columns:
        merged[nav_col] = merged[nav_col].fillna(method='ffill')

    # Drop rows where *both* the indicator and nav columns are still NaN after ffill
    merged = merged.dropna(subset=[indicator_col, nav_col], how='any')

    # Filter by plot_start_date and plot_end_date
    if plot_start_date and plot_end_date:
        start_dt = pd.to_datetime(plot_start_date)
        end_dt = pd.to_datetime(plot_end_date)
        merged = merged[(merged.index >= start_dt) & (merged.index <= end_dt)]

    if merged.empty:
        print(f"警告：在指定日期范围内数据为空或经过处理后数据不足，无法绘制图表: {title}")
        return

    # Create plot
    fig, ax1 = plt.subplots(figsize=figsize)

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
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

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

    # Add correlation coefficient
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
    # 设置日期范围
    start_date = '20100101'
    end_date = current_date
    report_start_date = '20170103'
    report_end_date = current_date
    
    print(f"开始日期: {start_date}, 结束日期: {end_date}")
    
    # 1. 获取申万一级行业列表
    industry_codes, industry_names = get_sw_industry_list()
    if not industry_codes:
        return
    
    # 2. 获取行业日行情数据
    print("正在获取行业日行情数据...")
    industry_data = get_cached_data(
        "industry_data.pkl", 
        get_industry_daily, 
        industry_codes, start_date, end_date
    )
    if industry_data.empty:
        print("未能获取行业日行情数据，程序终止。")
        return
    
    # 3. 获取沪深300净值数据
    print("正在获取沪深300指数数据...")
    hs300 = get_cached_data(
        "hs300_data.pkl",
        get_hs300_data,
        start_date, end_date
    )
    if hs300.empty:
        print("未能获取沪深300指数数据，程序终止。")
        return
    
    # 4. 计算各指标并绘制图表
    print("正在计算各指标并绘制图表...")
    
    # 图4：行业成交额占比一致性与沪深300净值
    print("计算图4指标: 行业成交额占比一致性...")
    corr_df, amt_corr_data = get_cached_data(
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
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )
    
    # 图5：行业成交额占比波动水平
    print("计算图5指标: 行业成交额占比波动水平...")
    if not corr_df.empty:
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
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )
        
    # 图6：A股市场交易的行业集中度
    print("计算图6指标: 行业集中度...")
    if amt_corr_data is not None and not amt_corr_data.empty and 'relative_amt' in amt_corr_data.columns:
        conc_df = get_cached_data(
            "industry_concentration.pkl",
            calculate_industry_concentration,
            amt_corr_data
        )
        if not conc_df.empty:
            merged_fig6 = pd.merge(conc_df, hs300, on='trade_date', how='left')
            corr_value_fig6 = calculate_correlation(merged_fig6, 'concentration_ma20', 'close', report_start_date, report_end_date)
            plot_chart(
                conc_df, hs300, 
                title='A股市场交易的行业集中度', 
                ylabel_left='行业集中度(MA20)', 
                ylabel_right='沪深300净值',
                indicator_col='concentration_ma20',
                corr_value=corr_value_fig6,
                ylim_left=(1.0, 3.5),
                ylim_right=(0.7, 1.9),
                plot_start_date=report_start_date,
                plot_end_date=report_end_date
            )
    
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
            ylim_left=(-0.1, 0.3),
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )
        
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
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )
        
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
            ylim_left=(0.036, 0.050),
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )
        
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
            ylim_left=(0, 90),
            ylim_right=(0.8, 1.8),
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )
        
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
            ylim_left=(0.1, 0.5),
            ylim_right=(0.6, 1.8),
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )
        
    print("所有图表生成完成！")

if __name__ == '__main__':
    main()
