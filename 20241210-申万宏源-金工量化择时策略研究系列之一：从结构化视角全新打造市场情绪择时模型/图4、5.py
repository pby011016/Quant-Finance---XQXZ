import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr, pearsonr
import time
import random

# 设置Tushare token
ts.set_token("4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77")
pro = ts.pro_api()

# 获取申万一级行业列表
def get_sw_industry_list():
    df = pro.index_classify(level='L1', src='SW2021')
    return df['index_code'].tolist()

# 获取行业日行情数据（带延时处理）
def get_industry_daily(industry_codes, start_date, end_date):
    all_data = []
    total_industries = len(industry_codes)
    
    for i, code in enumerate(industry_codes):
        try:
            print(f"获取行业数据 [{i+1}/{total_industries}]: {code}", end='')
            
            # 获取行业数据
            df = pro.sw_daily(ts_code=code, start_date=start_date, end_date=end_date, 
                                fields='trade_date,amount,float_mv')
            df['industry_code'] = code
            all_data.append(df)
            
            print(" - 成功")
            
            # 每5个请求后随机延时3-6秒，避免API限制
            if (i+1) % 5 == 0 and i < total_industries - 1:
                delay = random.uniform(3, 6)
                print(f"已完成 {i+1}/{total_industries}，等待 {delay:.1f} 秒...")
                time.sleep(delay)
                
        except Exception as e:
            print(f" - 失败: {e}")
            # 如果失败，等待更长时间后重试
            time.sleep(60)
            try:
                print(f"重试获取行业数据: {code}")
                df = pro.sw_daily(ts_code=code, start_date=start_date, end_date=end_date, 
                                    fields='trade_date,amount,float_mv')
                df['industry_code'] = code
                all_data.append(df)
                print(" - 重试成功")
            except:
                print(f" - 重试失败，跳过行业 {code}")
    
    return pd.concat(all_data)

# 计算行业成交额占比一致性指标
def calculate_amt_correlation(df):
    # 计算每个行业的相对成交额（成交额/流通市值）
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

# 获取沪深300指数数据
def get_hs300_data(start_date, end_date):
    df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date,
                        fields='trade_date,close')
    # 找到2017-01-03的数据作为基准
    start_value = df[df['trade_date'] == '20170103']['close'].values[0]
    df['nav'] = df['close'] / start_value
    return df[['trade_date', 'nav', 'close']]

# 计算相关系数（匹配报告方法）
def calculate_correlation(df, indicator_col, price_col, start_date, end_date):
    """
    计算指标与价格之间的相关系数
    :param df: 包含指标和价格的数据框
    :param indicator_col: 指标列名
    :param price_col: 价格列名
    :param start_date: 开始日期 (YYYYMMDD)
    :param end_date: 结束日期 (YYYYMMDD)
    :return: 相关系数
    """
    # 筛选日期范围
    filtered = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
    
    # 删除缺失值
    filtered = filtered.dropna(subset=[indicator_col, price_col])
    
    # 计算皮尔逊相关系数
    corr, _ = pearsonr(filtered[indicator_col], filtered[price_col])
    
    return corr

# 绘制图4：行业成交额占比一致性与沪深300净值
def plot_figure4(corr_df, hs300, corr_value):
    # 合并数据
    merged = pd.merge(corr_df, hs300, on='trade_date', how='left')
    
    # 转换为日期格式
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged = merged.set_index('trade_date').dropna()
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 绘制一致性指标 - 匹配报告中的蓝色
    ax1.plot(merged.index, merged['amt_corr_ma'], 
            label='行业成交额占比一致性(MA20)', 
            color='#1f77b4', linewidth=2.5)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('行业成交额占比一致性', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(0.8, 1.0)  # 匹配报告中图4的Y轴范围
    
    # 绘制沪深300净值 - 匹配报告中的橙色
    ax2 = ax1.twinx()
    ax2.plot(merged.index, merged['nav'], 
            label='沪深300净值', 
            color='#ff7f0e', alpha=0.9, linewidth=2.5)
    ax2.set_ylabel('沪深300净值', color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0.6, 1.8)  # 匹配报告中图4的净值范围
    
    # 设置日期格式 - 匹配报告的X轴格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # 添加标题 - 匹配报告标题
    plt.title('行业成交额占比一致性与沪深300净值', fontsize=14, fontweight='bold')
    
    # 添加图例 - 匹配报告位置
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # 添加网格 - 匹配报告样式
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # 添加数据源说明 - 匹配报告位置
    plt.figtext(0.85, 0.01, '资料来源：Tushare，申万宏源研究', ha='center', fontsize=9)
    
    # 添加注释 - 匹配报告中的"nav(右轴)"标注
    plt.text(0.02, 0.95, 'nav(右轴)', transform=ax2.transAxes, 
             color='#ff7f0e', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('figure4_industry_amt_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

# 绘制图5：行业成交额占比波动水平
def plot_figure5(corr_df, hs300, corr_value):
    # 合并数据
    merged = pd.merge(corr_df, hs300, on='trade_date', how='left')
    
    # 转换为日期格式
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged = merged.set_index('trade_date').dropna()
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 绘制波动率指标 - 匹配报告中的红色
    ax1.plot(merged.index, merged['amt_corr_std_ma'], 
            label='行业成交额占比波动水平(MA20)', 
            color='#d62728', linewidth=2.5)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('行业成交额占比波动水平', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_ylim(0.00, 0.10)  # 匹配报告中图5的Y轴范围
    
    # 绘制沪深300净值 - 匹配报告中的橙色
    ax2 = ax1.twinx()
    ax2.plot(merged.index, merged['nav'], 
            label='沪深300净值', 
            color='#ff7f0e', alpha=0.9, linewidth=2.5)
    ax2.set_ylabel('沪深300净值', color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0.6, 1.8)  # 匹配报告中图5的净值范围
    
    # 设置日期格式 - 匹配报告的X轴格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # 添加标题 - 匹配报告标题
    plt.title('行业成交额占比波动水平', fontsize=14, fontweight='bold')
    
    # 添加图例 - 匹配报告位置
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # 添加网格 - 匹配报告样式
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # 添加数据源说明 - 匹配报告位置
    plt.figtext(0.85, 0.01, '资料来源：Tushare，申万宏源研究', ha='center', fontsize=9)
    
    # 添加注释 - 匹配报告中的"nav(右轴)"标注
    plt.text(0.02, 0.95, 'nav(右轴)', transform=ax2.transAxes, 
             color='#ff7f0e', fontsize=10, verticalalignment='top')
    
    # 添加相关系数标注 - 匹配报告中的描述
    plt.figtext(0.15, 0.85, f"与沪深300价格指数的相关系数: {corr_value:.2f}", 
                ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure5_industry_amt_volatility.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
def main():
    # 设置日期范围 - 匹配报告中的时间范围
    start_date = '20170101'
    end_date = '20241115'
    
    # 1. 获取申万一级行业列表
    print("获取申万一级行业列表...")
    industry_codes = get_sw_industry_list()
    print(f"获取到{len(industry_codes)}个申万一级行业")
    
    # 2. 获取行业日行情数据（带延时）
    print("获取行业日行情数据（带延时处理）...")
    industry_data = get_industry_daily(industry_codes, start_date, end_date)
    print(f"成功获取{len(industry_data)}条行业行情数据")
    
    # 3. 计算行业成交额占比一致性指标
    print("计算行业成交额占比一致性指标...")
    corr_df, amt_df = calculate_amt_correlation(industry_data)
    
    # 4. 获取沪深300净值数据 - 以2017-01-03为基准
    print("获取沪深300指数数据...")
    hs300 = get_hs300_data(start_date, end_date)
    
    # 5. 计算相关系数（匹配报告方法）
    # 图5：行业成交额占比波动水平与沪深300价格之间的相关系数
    print("计算相关系数...")
    
    # 合并数据用于计算相关系数
    merged = pd.merge(corr_df, hs300, on='trade_date', how='left')
    merged = merged.dropna(subset=['amt_corr_std_ma', 'close'])
    
    # 计算图5的相关系数（行业成交额占比波动水平 vs 沪深300价格）
    # 报告中使用的是2010/3/5~2024/11/15，但我们的数据从2017年开始
    # 使用实际可用数据范围
    corr_value_fig5 = calculate_correlation(
        merged, 
        'amt_corr_std_ma',  # 波动水平指标
        'close',            # 沪深300收盘价
        '20170305',         # 报告中的起始日期
        '20241115'          # 报告中的结束日期
    )
    
    print(f"行业成交额占比波动水平与沪深300价格相关系数: {corr_value_fig5:.4f}")
    
    # 图4：行业成交额占比一致性与沪深300净值
    # 报告中未给出具体相关系数，但我们可以计算
    merged = pd.merge(corr_df, hs300, on='trade_date', how='left')
    merged = merged.dropna(subset=['amt_corr_ma', 'close'])
    corr_value_fig4 = calculate_correlation(
        merged, 
        'amt_corr_ma',      # 一致性指标
        'close',            # 沪深300收盘价
        '20170305',         # 报告中的起始日期
        '20241115'          # 报告中的结束日期
    )
    print(f"行业成交额占比一致性与沪深300价格相关系数: {corr_value_fig4:.4f}")
    
    # 6. 绘制图4 - 匹配报告中的图4
    print("绘制图4：行业成交额占比一致性与沪深300净值...")
    plot_figure4(corr_df, hs300, corr_value_fig4)
    
    # 7. 绘制图5 - 匹配报告中的图5
    print("绘制图5：行业成交额占比波动水平...")
    plot_figure5(corr_df, hs300, corr_value_fig5)
    
    print("图表生成完成！")

if __name__ == '__main__':
    main()