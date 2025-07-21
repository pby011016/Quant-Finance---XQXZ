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

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置当前日期为最新
current_date = datetime.now().strftime("%Y%m%d")

# 忽略警告
warnings.filterwarnings('ignore')

# 设置Tushare token
ts.set_token("4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77")
pro = ts.pro_api()

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
                              fields='trade_date,amount,float_mv,close,pct_change')
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
                
        except Exception as e:
            print(f" - 获取行业 {code} 数据失败: {e}")
            time.sleep(30)
    
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
    corr_df['amt_corr_ma'] = corr_df['amt_corr'].rolling(20).mean()  # 修正列名
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
        
        # 获取换手率前五的行业
        top5 = group.nlargest(5, 'relative_amt')
        top5_avg = top5['relative_amt'].mean()
        
        # 计算全市场平均换手率
        market_avg = group['relative_amt'].mean()
        
        # 计算集中度
        if market_avg == 0:
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
    """创业板成交活跃度"""
    print("正在获取创业板成交活跃度数据...")
    
    # 1. 获取创业板成交额（399006.SZ）
    gem = pro.index_daily(ts_code='399006.SZ', 
                          start_date=start_date, 
                          end_date=end_date,
                          fields='trade_date,amount')
    if gem is None or gem.empty:
        print("警告：未获取到创业板成交额数据")
        return pd.DataFrame()
    gem.columns = ['trade_date', 'amount_gem']
    
    # 2. 获取全市场A股成交额（使用daily表，包含所有股票）
    all_a = pro.daily(trade_date='', 
                      start_date=start_date, 
                      end_date=end_date,
                      fields='trade_date,amount')
    if all_a is None or all_a.empty:
        print("警告：未获取到全市场成交额数据")
        return pd.DataFrame()
    all_a = all_a.groupby('trade_date')['amount'].sum().reset_index()
    all_a.columns = ['trade_date', 'amount_all']
    
    # 3. 外连接合并数据（保留所有日期）
    merged = pd.merge(gem, all_a, on='trade_date', how='outer')
    
    # 4. 计算创业板成交活跃度（处理除零和缺失值）
    merged['gem_activity'] = merged['amount_gem'] / merged['amount_all'].replace(0, np.nan)
    merged['gem_activity'] = merged['gem_activity'].fillna(method='ffill')  # 向前填充
    merged = merged.dropna(subset=['gem_activity'])
    
    # 5. 计算20日均线（报告中使用20日均线）
    merged['gem_activity_ma20'] = merged['gem_activity'].rolling(20).mean()
    
    # 6. 筛选报告时间范围（2017-2024）
    report_start = pd.to_datetime('20170103')
    merged = merged[pd.to_datetime(merged['trade_date']) >= report_start]
    
    return merged[['trade_date', 'gem_activity', 'gem_activity_ma20']]

# 图9：融资余额占流通市值比
def calculate_margin_ratio(start_date, end_date):
    print("计算融资余额占比...")
    
    margin = pro.margin(start_date=start_date, end_date=end_date, fields='trade_date,rzye')
    if margin.empty:
        return pd.DataFrame()

    # 存储结果
    offset = 0
    all_data = []
    while True:
        df = pro.daily_basic(start_date=start_date, end_date=end_date, fields='trade_date,circ_mv', limit=6000, offset=offset)
        if df.empty:
            break
        all_data.append(df)
        offset += 6000
        time.sleep(0.5)  # 避免请求过快

    # 合并所有分页数据
    market_val = pd.concat(all_data, ignore_index=True)
    market_val.sort_values('trade_date', inplace=True)

    if market_val.empty:
        return pd.DataFrame()

    # 统一单位：rzye 是元，circ_mv 是万元
    market_val['circ_mv'] = market_val['circ_mv'] * 10000

    # 合并数据并按日期排序
    merged = pd.merge(margin, market_val, on='trade_date', how='inner')
    merged = merged.sort_values('trade_date')

    merged['margin_ratio'] = merged['rzye'] / merged['circ_mv'].replace(0, np.nan)
    merged['margin_ratio'] = merged['margin_ratio'].fillna(method='ffill')
    
    # 计算60日均线
    merged['margin_ratio_ma60'] = merged['margin_ratio'].rolling(60).mean()
    
    return merged[['trade_date', 'margin_ratio', 'margin_ratio_ma60']]


def calculate_industry_rotation(industry_data):
    """计算行业轮涨补涨程度指标"""
    if industry_data.empty:
        return pd.DataFrame()

    # 直接使用涨跌幅
    # 按日期和行业排序
    industry_data = industry_data.sort_values(['trade_date', 'industry_code'])
    
    # 计算轮动相关性
    rotation_data = []
    dates = sorted(industry_data['trade_date'].unique())
    
    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i-1]
        
        # 获取当前日期和前一日期的行业涨跌幅
        current_data = industry_data[industry_data['trade_date'] == current_date][['industry_code', 'pct_change']]
        prev_data = industry_data[industry_data['trade_date'] == prev_date][['industry_code', 'pct_change']]
        
        # 合并数据
        merged = pd.merge(prev_data, current_data, on='industry_code', suffixes=('_prev', '_curr'), how='inner')
        
        if len(merged) < 2:
            rotation_data.append({'trade_date': current_date, 'rotation_corr': np.nan})
            continue

        # 直接使用涨跌幅计算排名
        merged['rank_prev'] = merged['pct_chg_prev'].rank(ascending=False, method='dense')
        merged['rank_curr'] = merged['pct_chg_curr'].rank(ascending=False, method='dense')
        
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
    
    # 计算平均增益和平均损失
    hs300['avg_gain'] = hs300['gain'].rolling(window=window, min_periods=1).mean()
    hs300['avg_loss'] = hs300['loss'].rolling(window=window, min_periods=1).mean()
    
    # 计算相对强度 (RS)
    hs300['rs'] = hs300['avg_gain'] / hs300['avg_loss'].replace(0, np.nan)
    
    # 计算RSI
    hs300['rsi'] = 100 - (100 / (1 + hs300['rs']))
    
    # 计算20日均线
    hs300['rsi_ma20'] = hs300['rsi'].rolling(20).mean()
    
    return hs300[['trade_date', 'rsi', 'rsi_ma20']]


def get_hs300_data(start_date, end_date):
    """获取沪深300指数数据"""
    print("正在获取沪深300指数数据...")
    df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date,
                         fields='trade_date,close')
    
    if df is None or df.empty:
        return pd.DataFrame()

    # 确保数据按日期排序
    df = df.sort_values('trade_date')
    
    # 找到基准日期的数据
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
        return np.nan

    # 删除缺失值
    filtered = filtered.dropna(subset=[indicator_col, price_col])
    
    if len(filtered) < 2:
        return np.nan
    
    # 计算皮尔逊相关系数
    corr, _ = pearsonr(filtered[indicator_col], filtered[price_col])
    
    return corr

def plot_chart(data, hs300, title, ylabel_left, ylabel_right, 
               indicator_col, nav_col='nav', corr_value=None,
               ylim_left=None, ylim_right=None, figsize=(14, 8),
               plot_start_date=None, plot_end_date=None):
    """绘制指标与沪深300净值的双轴图"""
    # 合并数据
    merged = pd.merge(data, hs300, on='trade_date', how='inner')
    
    # 转换为日期格式
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged = merged.set_index('trade_date').dropna()
    
    if merged.empty:
        print(f"警告：合并后的数据为空，无法绘制图表: {title}")
        return

    # 根据日期筛选数据
    if plot_start_date and plot_end_date:
        merged = merged[(merged.index >= pd.to_datetime(plot_start_date)) & 
                        (merged.index <= pd.to_datetime(plot_end_date))]
    
    if merged.empty:
        return

    # 创建图表
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 绘制指标
    ax1.plot(merged.index, merged[indicator_col], 
             label=ylabel_left, 
             color='#1f77b4', linewidth=2.5)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel(ylabel_left, color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    if ylim_left:
        ax1.set_ylim(ylim_left)
    
    # 绘制沪深300净值
    ax2 = ax1.twinx()
    ax2.plot(merged.index, merged[nav_col], 
             label=ylabel_right, 
             color='#ff7f0e', alpha=0.9, linewidth=2.5)
    ax2.set_ylabel(ylabel_right, color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    if ylim_right:
        ax2.set_ylim(ylim_right)
    
    # 设置日期格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator()) 
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    # 添加标题
    plt.title(title, fontsize=14, fontweight='bold')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # 添加数据来源
    plt.figtext(0.85, 0.01, '资料来源：Tushare，申万宏源研究', ha='center', fontsize=9)
    
    # 添加相关系数
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
        return
    
    # 3. 获取沪深300净值数据
    print("正在获取沪深300指数数据...")
    hs300 = get_cached_data(
        "hs300_data.pkl",
        get_hs300_data,
        start_date, end_date
    )
    if hs300.empty:
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
            indicator_col='amt_corr_ma',  # 修正列名
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
        gem_activity_df = get_cached_data(
            "gem_activity.pkl",
            calculate_gem_activity,
            start_date, end_date
        )
        if not gem_activity_df.empty:
            merged_fig8 = pd.merge(gem_activity_df, hs300, on='trade_date', how='inner')
            corr_value_fig8 = calculate_correlation(merged_fig8, 'gem_activity_ma20', 'close', report_start_date, report_end_date)
            plot_chart(
                gem_activity_df, 
                hs300, 
                title='创业板成交活跃度与沪深300净值', 
                ylabel_left='创业板成交额/全A成交额(MA20)', 
                ylabel_right='沪深300净值',
                indicator_col='gem_activity_ma20',
                corr_value=corr_value_fig8,
                ylim_left=(0.05, 0.35),
                ylim_right=(0.6, 1.8),
                plot_start_date=report_start_date,
                plot_end_date=report_end_date
            )

    # # 图9：融资余额占自由流通市值比
    # print("计算图9指标: 融资余额占自由流通市值比...")
    # margin_ratio_df = calculate_margin_ratio(start_date, end_date)
    # merged = pd.merge(margin_ratio_df, hs300, on='trade_date', how='left')
    # corr_value_fig9 = calculate_correlation(merged, 'margin_ratio_ma60', 'close', report_start_date, report_end_date)
    # plot_chart(
    #     margin_ratio_df, hs300, 
    #     title='融资余额占自由流通市值比', 
    #     ylabel_left='融资余额占比(MA60)', 
    #     ylabel_right='沪深300净值',
    #     indicator_col='margin_ratio_ma60',
    #     corr_value=corr_value_fig9,
    #     ylim_left=(0.035, 0.048),
    #     ylim_right=(0.6, 1.8)
    # )
    
        
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
        "industry_rotation_corrected.pkl",
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
            ylim_left=(-0.2, 0.5),
            ylim_right=(0, 2),
            plot_start_date=report_start_date,
            plot_end_date=report_end_date
        )

   
    print("所有图表生成完成！")

if __name__ == '__main__':
    main()