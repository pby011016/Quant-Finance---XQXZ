# -*- coding: utf-8 -*-
"""完整图14-图21复现代码"""
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.optimize import brentq
import os
import pickle
import threading
import time
import warnings
from datetime import datetime, timedelta
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置当前日期
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

def black_scholes_iv(S, K, T, r, option_type, price):
    """使用Black-Scholes模型计算隐含波动率"""
    def black_scholes(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - price
        else:  # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - price
    
    try:
        # 使用布伦特方法求解隐含波动率
        return brentq(black_scholes, 1e-6, 5.0, xtol=1e-4)
    except:
        return np.nan

def calculate_vix_for_date(option_data, underlying_price, trade_date):
    """计算某一交易日的VIX指数"""
    today = pd.to_datetime(trade_date)
    
    # 筛选出两个最近到期月份
    expiries = option_data['maturity_date'].unique()
    if len(expiries) < 2:
        return np.nan
    
    # 按距离排序到期日
    expiries.sort()
    near_expiry = expiries[0]
    next_expiry = expiries[1]
    
    # 计算两个到期日的时间（年）
    T1 = (near_expiry - today).days / 365.0
    T2 = (next_expiry - today).days / 365.0
    
    # 筛选出两个到期日的看涨和看跌期权
    near_options = option_data[option_data['maturity_date'] == near_expiry]
    next_options = option_data[option_data['maturity_date'] == next_expiry]
    
    # 找到最接近平价的期权（执行价最接近标的当前价格）
    def find_atm_options(options):
        if options.empty:
            return pd.DataFrame()
        options['moneyness'] = abs(options['exercise_price'] - underlying_price)
        return options.sort_values('moneyness').head(4)  # 每个月份取4个（2个call，2个put）
    
    near_atm = find_atm_options(near_options)
    next_atm = find_atm_options(next_options)
    
    if near_atm.empty or next_atm.empty:
        return np.nan
    
    # 计算隐含波动率
    near_atm['iv'] = near_atm.apply(
        lambda row: black_scholes_iv(
            underlying_price, row['exercise_price'], T1, 0.03, row['call_put'], row['close']
        ), axis=1
    )
    next_atm['iv'] = next_atm.apply(
        lambda row: black_scholes_iv(
            underlying_price, row['exercise_price'], T2, 0.03, row['call_put'], row['close']
        ), axis=1
    )
    
    # 计算两个期限的平均隐含波动率
    near_avg_iv = near_atm['iv'].mean()
    next_avg_iv = next_atm['iv'].mean()
    
    # 计算VIX（加权平均）
    w = (T2 - 30/365) / (T2 - T1)  # 权重因子，假设30天为基准
    vix = w * near_avg_iv + (1 - w) * next_avg_iv
    return vix * 100  # 转换为百分比形式

def get_option_data(ts_code, start_date, end_date):
    """获取期权数据（包括基础信息和日行情）"""
    # 获取期权合约基础信息
    basic = pro.opt_basic(exchange='SSE', fields='ts_code,name,exercise_price,call_put,list_date,delist_date,maturity_date')
    
    # 过滤标的
    if '50' in ts_code:
        basic = basic[basic['name'].str.contains('50ETF')]
    else:
        basic = basic[basic['name'].str.contains('沪深300')]
    
    if basic.empty:
        return pd.DataFrame()
    


    # 获取期权日行情
    all_daily = []
    # for code in tqdm(basic['ts_code'], desc="下载期权日线数据"):
    #     try:
    #         daily = pro.opt_daily(ts_code=code, start_date=start_date, end_date=end_date, fields='trade_date,ts_code,close')
    #         all_daily.append(daily)
    #         # break
    #     except:
    #         print("error")
    #         time.sleep(2)  # 避免请求过快
    #         continue
    
    start_date_dt = datetime.strptime(start_date, "%Y%m%d")
    end_date_dt = datetime.strptime(end_date, "%Y%m%d")
    current_date = start_date_dt
    while current_date <= end_date_dt:
        try:
            # 格式化为YYYYMMDD字符串
            date_str = current_date.strftime("%Y%m%d")
            daily = pro.opt_daily(exchange='SSE', start_date=date_str, end_date=date_str, fields='trade_date,ts_code,close')
            all_daily.append(daily)

            # 日期递增
            current_date += timedelta(days=1)
        except:
            print("time sleep 15s")
            time.sleep(15)
            continue
    
    if not all_daily:
        return pd.DataFrame()
    
    daily_df = pd.concat(all_daily, ignore_index=True)
    
    # 合并基础信息
    merged = pd.merge(daily_df, basic, on='ts_code')
    
    # 转换日期
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged['maturity_date'] = pd.to_datetime(merged['maturity_date'])
    
    # 过滤标的
    if '50' in ts_code:
        merged = merged[merged['name'].str.contains('50ETF')]
    else:
        merged = merged[merged['name'].str.contains('沪深300')]
    
    return merged[['trade_date', 'ts_code', 'close', 'exercise_price', 'call_put', 'maturity_date']]

def get_vix_data(ts_code, start_date, end_date):
    """计算VIX指数"""
    print(f"开始计算 {ts_code} 的VIX指数，日期范围：{start_date} 至 {end_date}")
    
    # 获取标的物价格
    if '50' in ts_code:
        # 50ETF
        underlying_code = '510050.SH'
        underlying_data = pro.fund_daily(ts_code=underlying_code, 
                                        start_date=start_date, 
                                        end_date=end_date,
                                        fields='trade_date,close')
    else:
        # 沪深300ETF（上交所）
        underlying_code = '510300.SH'
        underlying_data = pro.fund_daily(ts_code=underlying_code,
                                        start_date=start_date,
                                        end_date=end_date,
                                        fields='trade_date,close')
    
    print(f"获取到 {len(underlying_data)} 条标的物数据")
    
    underlying_data['trade_date'] = pd.to_datetime(underlying_data['trade_date'])
    underlying_data = underlying_data.set_index('trade_date')['close']
    
    # 获取期权数据
    option_data = get_option_data(ts_code, start_date, end_date)
    print(f"获取到 {len(option_data)} 条期权数据")
    
    if option_data.empty:
        print("警告：没有获取到期权数据")
        return pd.DataFrame()
    
    # 计算每日VIX
    vix_values = []
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for trade_date in dates:
            if trade_date in underlying_data.index:
                print(f"\n处理日期：{trade_date.strftime('%Y%m%d')}")
                underlying_price = underlying_data.loc[trade_date]
                print(f"标的物价格：{underlying_price}")
                
                date_options = option_data[option_data['trade_date'] == trade_date]
                print(f"该日期有 {len(date_options)} 个期权合约")
                
                if len(date_options) >= 8:
                    futures.append(executor.submit(
                        calculate_vix_for_date, date_options, underlying_price, trade_date
                    ))
                else:
                    print("警告：合约数量不足8个，跳过计算")
        
        for i, future in enumerate(futures):
            if i % 10 == 0:
                print(f"计算VIX进度: {i+1}/{len(futures)}")
            try:
                vix = future.result()
                if vix is not None:
                    vix_values.append({
                        'trade_date': dates[i].strftime("%Y%m%d"),
                        'vix': vix
                    })
            except Exception as e:
                print(f"计算VIX时出错：{str(e)}")
    
    if not vix_values:
        print("警告：未能计算出任何VIX值")
        return pd.DataFrame()
    
    return pd.DataFrame(vix_values)

def get_50etf_vix_data(start_date, end_date):
    """获取50ETF波指数据"""
    return get_cached_data(
        f"vix_50etf_{start_date}_{end_date}.pkl",
        get_vix_data,
        '50ETF', start_date, end_date
    )

def get_300_vix_data(start_date, end_date):
    """获取300波指数据"""
    return get_cached_data(
        f"vix_300_{start_date}_{end_date}.pkl",
        get_vix_data,
        '300', start_date, end_date
    )

def get_50etf_price(start_date, end_date):
    """获取50ETF价格数据"""
    df = pro.fund_daily(ts_code='510050.SH', start_date=start_date, end_date=end_date,
                       fields='trade_date,close')
    df = df.rename(columns={'close': 'price_50etf'})
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
    return df[['trade_date', 'price_50etf']]

def get_hs300_index(start_date, end_date):
    """获取沪深300指数数据"""
    df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date,
                        fields='trade_date,close')
    df = df.rename(columns={'close': 'price_hs300'})
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
    return df[['trade_date', 'price_hs300']]

class TushareRateController:
    """Tushare API请求速率控制器"""
    def __init__(self, max_per_minute=140):
        """
        :param max_per_minute: 每分钟最大请求数
        """
        self.max_requests = max_per_minute
        self.request_times = []
        self.lock = threading.Lock()
    
    def wait_for_slot(self):
        """等待可用的请求名额"""
        with self.lock:
            # 清理超时请求记录（60秒窗口）
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            # 检查当前请求数
            if len(self.request_times) >= self.max_requests:
                # 计算最早请求的剩余时间
                oldest_time = self.request_times[0]
                wait_time = 60 - (now - oldest_time)
                if wait_time > 0:
                    time.sleep(wait_time + 0.5)  # 额外0.5秒缓冲
                    # 更新请求时间列表
                    self.request_times = [t for t in self.request_times if now - t < 60]
            
            # 记录本次请求时间
            self.request_times.append(time.time())

def get_300_option_data_corrected(start_date, end_date):
    """严格限速的沪深300期权数据获取"""
    if os.path.exists("300_option_data"):
        with open("300_option_data", 'rb') as f:
            all_data = pickle.load(f)
    else:
        # 初始化速率控制器（留10%缓冲空间）
        rate_controller = TushareRateController(max_per_minute=135)
        
        # 获取合约基本信息（含call_put字段）
        contracts = pro.opt_basic(
            exchange='CFFEX', 
            opt_type='I',
            fields='ts_code,call_put'
        )
        print(f"获取到 {len(contracts)} 个沪深300股指期权合约")
        
        # 按到期月份分组处理（减少单次请求压力）
        contracts['expiry_month'] = contracts['ts_code'].apply(lambda x: x.split('-')[1])
        grouped_contracts = contracts.groupby('expiry_month')
        
        all_data = []
        processed_contracts = 0
        total_contracts = len(contracts)
        
        # 分批处理合约组
        for month, group in grouped_contracts:
            print(f"处理 {month} 月份合约组 ({len(group)} 个合约)")

            max_retries = 3  # 最大重试次数
            retry_delay = 60  # 重试等待时间（秒）

            for _, row in group.iterrows():
                ts_code = row['ts_code']
                call_put = row['call_put']
                retry_count = 0
                success = False  # 标记是否成功获取数据
                
                while retry_count < max_retries and not success:
                    # 等待请求名额
                    rate_controller.wait_for_slot()
                    
                    # 尝试获取合约数据
                    try:
                        df = pro.opt_daily(
                            ts_code=ts_code,
                            start_date=start_date,
                            end_date=end_date,
                            fields='trade_date,vol'
                        )
                        # 添加期权类型
                        df['call_put'] = call_put
                        df['contract'] = ts_code
                        all_data.append(df)
                        success = True
                        
                        # 进度跟踪
                        processed_contracts += 1
                        if processed_contracts % 10 == 0:
                            print(f"进度: {processed_contracts}/{total_contracts} | "
                                f"剩余配额: {135 - len(rate_controller.request_times)}")
                        
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        if "每分钟最多访问" in error_msg:
                            print(f"合约 {ts_code} 触发API限制 ({retry_count}/{max_retries})，等待 {retry_delay}秒...")
                            time.sleep(retry_delay)
                        else:
                            print(f"获取合约 {ts_code} 失败: {error_msg}")
                            time.sleep(10)
                
                # 最终重试失败处理
                if not success:
                    print(f"⚠️ 放弃合约 {ts_code} (重试 {max_retries} 次仍失败)")
            

        # 合并数据并计算PCR
        if not all_data:
            return pd.DataFrame()
        
        with open("300_option_data", 'wb') as f:
            pickle.dump(all_data, f)
    
    full_data = pd.concat(all_data)
    return calculate_pcr_corrected(full_data)

def calculate_pcr_corrected(df):
    """
    1. 分离认购/认沽合约
    2. 按日聚合成交量
    3. 计算PCR
    """
    if df.empty:
        return pd.DataFrame()
    
    # 分离认购合约 (call_put='C')
    call_df = df[df['call_put'] == 'C'].groupby('trade_date')['vol'].sum().reset_index()
    call_df.rename(columns={'vol': 'call_vol'}, inplace=True)
    
    # 分离认沽合约 (call_put='P')
    put_df = df[df['call_put'] == 'P'].groupby('trade_date')['vol'].sum().reset_index()
    put_df.rename(columns={'vol': 'put_vol'}, inplace=True)
    
    # 合并数据
    merged = pd.merge(call_df, put_df, on='trade_date', how='outer').fillna(0)
    
    # 计算PCR
    merged['pcr'] = merged.apply(
        lambda x: x['put_vol'] / x['call_vol'] if x['call_vol'] > 0 else float('inf'),
        axis=1
    )
    # 确保日期统一为相同格式（字符串或datetime）
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    # 确保日期连续性
    date_range = pd.date_range(
        start=merged['trade_date'].min(), 
        end=merged['trade_date'].max()
    )
    merged = merged.set_index('trade_date').reindex(date_range).fillna(method='ffill').reset_index()
    merged.rename(columns={'index': 'trade_date'}, inplace=True)
    
    # 计算20日均线
    merged['pcr_ma20'] = merged['pcr'].rolling(20, min_periods=5).mean()
    
    return merged


def plot_dual_axis(data1, data2, title, ylabel1, ylabel2, 
                   col1, col2, figsize=(14, 8), 
                   color1='#1f77b4', color2='#ff7f0e'):
    """绘制双Y轴图表"""
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 绘制第一条线
    ax1.plot(data1['trade_date'], data1[col1], 
             label=ylabel1, 
             color=color1, linewidth=2.5)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel(ylabel1, color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 绘制第二条线
    ax2 = ax1.twinx()
    ax2.plot(data2['trade_date'], data2[col2], 
             label=ylabel2, 
             color=color2, alpha=0.9, linewidth=2.5)
    ax2.set_ylabel(ylabel2, color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 设置标题和格式
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.figtext(0.85, 0.01, '资料来源：Tushare，申万宏源研究', ha='center', fontsize=9)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    filename = title.replace(' ', '_').replace(':', '').replace('/', '_') + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {filename}")
    plt.show()
    plt.close()

def plot_three_lines(data, title, ylabels, cols, colors, figsize=(14, 8)):
    """绘制三条线的图表"""
    plt.figure(figsize=figsize)
    
    for i in range(3):
        plt.plot(data['trade_date'], data[cols[i]], 
                 label=ylabels[i], 
                 color=colors[i], linewidth=2.5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.figtext(0.85, 0.01, '资料来源：Tushare，申万宏源研究', ha='center', fontsize=9)
    
    plt.tight_layout()
    filename = title.replace(' ', '_').replace(':', '').replace('/', '_') + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存为: {filename}")
    plt.show()
    plt.close()



# 主程序
def main():
    # 设置日期范围（期权数据从2019年12月开始）
    start_date = '20191223'
    end_date = current_date
    end_date = '20250626'
    
    print(f"开始日期: {start_date}, 结束日期: {end_date}")

    # 图14：50ETF波指与标的价格
    print("生成图14：50ETF波指与标的价格...")
    vix_50 = get_50etf_vix_data(start_date, end_date)
    etf_50 = get_50etf_price(start_date, end_date)
    
    if not vix_50.empty and not etf_50.empty:
        # 确保日期格式一致
        vix_50['trade_date'] = pd.to_datetime(vix_50['trade_date'])
        etf_50['trade_date'] = pd.to_datetime(etf_50['trade_date'])
        
        # 统一按升序排序
        vix_50 = vix_50.sort_values('trade_date')
        etf_50 = etf_50.sort_values('trade_date')
        
        # 确保日期范围一致
        start = max(vix_50['trade_date'].min(), etf_50['trade_date'].min())
        end = min(vix_50['trade_date'].max(), etf_50['trade_date'].max())
        
        vix_50 = vix_50[(vix_50['trade_date'] >= start) & (vix_50['trade_date'] <= end)]
        etf_50 = etf_50[(etf_50['trade_date'] >= start) & (etf_50['trade_date'] <= end)]
        
        # 再次检查数据是否为空
        if not vix_50.empty and not etf_50.empty:
            plot_dual_axis(
                vix_50, etf_50,
                title='50ETF波指与标的价格',
                ylabel1='50ETF波指(VIX)',
                ylabel2='50ETF价格',
                col1='vix',
                col2='price_50etf'
            )
        else:
            print("警告：日期对齐后数据为空，无法绘制图表")
    else:
        print("警告：获取到的VIX或ETF数据为空，无法绘制图表")

    # 图15：300波指与标的价格
    print("生成图15：300波指与标的价格...")
    vix_300 = get_300_vix_data(start_date, end_date)
    hs300_price = get_hs300_index(start_date, end_date)
    
    if not vix_300.empty and not hs300_price.empty:
        # 确保日期列是datetime类型
        vix_300['trade_date'] = pd.to_datetime(vix_300['trade_date'])
        hs300_price['trade_date'] = pd.to_datetime(hs300_price['trade_date'])
        
        # 统一按升序排序
        vix_300 = vix_300.sort_values('trade_date')
        hs300_price = hs300_price.sort_values('trade_date')
        
        # 确保日期范围一致
        start = max(vix_300['trade_date'].min(), hs300_price['trade_date'].min())
        end = min(vix_300['trade_date'].max(), hs300_price['trade_date'].max())
        
        vix_300 = vix_300[(vix_300['trade_date'] >= start) & (vix_300['trade_date'] <= end)]
        hs300_price = hs300_price[(hs300_price['trade_date'] >= start) & (hs300_price['trade_date'] <= end)]
        
        # 再次检查数据是否为空
        if not vix_300.empty and not hs300_price.empty:
            plot_dual_axis(
                vix_300, hs300_price,
                title='300波指与标的价格',
                ylabel1='300波指(VIX)',
                ylabel2='沪深300指数价格',
                col1='vix',
                col2='price_hs300'
            )
        else:
            print("警告：日期对齐后数据为空，无法绘制图表")
    else:
        print("警告：获取到的VIX或指数数据为空，无法绘制图表")

    # 图16：50ETF波指与标的价格3M滚动相关系数
    print("生成图16：50ETF波指与标的价格3M滚动相关系数...")
    if not vix_50.empty and not etf_50.empty:
        # 合并数据
        merged_50 = pd.merge(vix_50, etf_50, on='trade_date').sort_values('trade_date')
        # 处理缺失值
        merged_50 = merged_50.dropna(subset=['vix', 'price_50etf']) 

        # 计算滚动相关系数（动态窗口）
        window_size = min(63, len(merged_50))
        merged_50['roll_corr'] = merged_50['vix'].rolling(window=window_size).corr(merged_50['price_50etf'])

        # 数据检查
        print("有效数据点:", len(merged_50['roll_corr'].dropna()))
        print("相关系数范围:", merged_50['roll_corr'].min(), merged_50['roll_corr'].max())
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # 红色相关系数线（左轴）
        ax1.plot(merged_50['trade_date'], merged_50['roll_corr'], 
                color='tab:red', linewidth=3, label='3M滚动相关系数')
        ax1.set_ylabel('滚动相关系数', fontsize=13, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_ylim(-1.1, 1.1)  # 固定相关系数范围
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        # 蓝色价格线（右轴）
        ax2 = ax1.twinx()
        ax2.plot(merged_50['trade_date'], merged_50['price_50etf'], 
                color='tab:blue', linewidth=1.5, alpha=0.8, label='50ETF价格')
        ax2.set_ylabel('50ETF价格', fontsize=13, color='tab:blue')
        
        # 图例合并
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
        
        # 其他格式设置
        plt.title('50ETF波指与标的价格3M滚动相关系数', fontsize=15, pad=20)
        ax1.grid(True, linestyle=':', alpha=0.6)
        plt.figtext(0.85, 0.01, '数据来源：Tushare | 申万宏源研究', ha='center', fontsize=10)
        plt.tight_layout()
        
        plt.savefig('50ETF波指与标的价格3M滚动相关系数.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 图17：300波指与标的价格3M滚动相关系数
    print("生成图17：300波指与标的价格3M滚动相关系数...")
    if not vix_300.empty and not hs300_price.empty:
        # 数据合并与预处理
        merged_300 = pd.merge(vix_300, hs300_price, on='trade_date').sort_values('trade_date')
        
        # 动态计算窗口（63个交易日≈3个月）
        window_size = min(63, len(merged_300) - 1)  # 确保至少window_size+1个数据点
        
        # 计算滚动相关系数（带容错处理）
        merged_300['roll_corr'] = merged_300['vix'].rolling(
            window=window_size,
            min_periods=max(3, int(window_size*0.5))  # 至少50%数据点
        ).corr(merged_300['price_hs300'])
        
        # 数据有效性检查
        valid_corr = merged_300['roll_corr'].dropna()
        if len(valid_corr) == 0:
            print("警告：未能计算出有效相关系数")
        else:
            # 创建双坐标轴图表
            fig, ax1 = plt.subplots(figsize=(16, 8))
            
            # 红色相关系数线（左轴）
            ax1.plot(merged_300['trade_date'], merged_300['roll_corr'], 
                    color='tab:red', linewidth=2.5, label='3M滚动相关系数')
            ax1.set_ylabel('滚动相关系数', fontsize=13, color='tab:red')
            ax1.set_ylim(-1.1, 1.1)  # 固定相关系数范围
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            # 蓝色价格线（右轴）
            ax2 = ax1.twinx()
            ax2.plot(merged_300['trade_date'], merged_300['price_hs300'], 
                    color='tab:blue', linewidth=1.2, alpha=0.7, label='沪深300指数')
            ax2.set_ylabel('指数价格', fontsize=13, color='tab:blue')
            
            # 图表装饰
            plt.title('300波指与标的价格3M滚动相关系数', fontsize=15, pad=20, fontweight='bold')
            
            # 合并图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
            
            # 添加数据来源
            plt.figtext(0.85, 0.01, '数据来源：Tushare | 申万宏源研究', 
                    ha='center', fontsize=10, color='gray')
            
            plt.tight_layout()
            plt.savefig('300波指与标的价格3M滚动相关系数.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()

    # 图18：中金所300指数PCR与沪深300指数
    print("生成图18：中金所300指数PCR与沪深300指数...")
    
    # 获取PCR数据
    option_data = get_300_option_data_corrected(start_date, end_date)
    
    # 获取沪深300指数数据
    hs300_data = get_hs300_index(start_date, end_date)
    
    if option_data.empty or hs300_data.empty:
        print("缺少必要数据，无法绘制图18")
        return
    
    # 数据处理
    option_data['trade_date'] = pd.to_datetime(option_data['trade_date'])
    hs300_data['trade_date'] = pd.to_datetime(hs300_data['trade_date'])
    
    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # PCR数据（左轴）
    color = 'tab:blue'
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('PCR(20日均线)', color=color, fontsize=12)
    ax1.plot(option_data['trade_date'], option_data['pcr_ma20'], 
             color=color, linewidth=2, label='PCR(20日均线)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 沪深300指数（右轴）
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('沪深300指数', color=color, fontsize=12)
    ax2.plot(hs300_data['trade_date'], hs300_data['price_hs300'], 
             color=color, linewidth=1.5, alpha=0.7, label='沪深300指数')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 图表装饰
    plt.title('中金所300指数PCR(20日均线)与沪深300指数', fontsize=14, pad=15)
    fig.tight_layout()
    plt.figtext(0.85, 0.01, '数据来源：Tushare | 申万宏源研究', ha='center', fontsize=9)
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # 保存图表
    plt.savefig('图18_中金所300指数PCR与沪深300指数.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图18已保存")

    # 图19：中金所300 PCR与认沽、认购成交量变化情况
    print("生成图19：中金所300 PCR与认沽、认购成交量变化情况...")

    # 准备数据
    option_data['call_vol_ma20'] = option_data['call_vol'].rolling(window=20).mean()
    option_data['put_vol_ma20'] = option_data['put_vol'].rolling(window=20).mean()
    option_data['total_vol_ma20'] = option_data['call_vol_ma20'] + option_data['put_vol_ma20']

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 认购成交量（左轴）
    color = 'tab:red'
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('成交量(20日均线, 万手)', color='k', fontsize=12)
    ax1.plot(option_data['trade_date'], option_data['call_vol_ma20'], 
            color=color, linewidth=2, label='认购成交量')
    ax1.plot(option_data['trade_date'], option_data['put_vol_ma20'], 
            color='tab:orange', linewidth=2, label='认沽成交量')
    ax1.plot(option_data['trade_date'], option_data['total_vol_ma20'], 
            color='tab:green', linewidth=2, label='总成交量')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    # PCR数据（右轴）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('PCR(20日均线)', color=color, fontsize=12)
    ax2.plot(option_data['trade_date'], option_data['pcr_ma20'], 
            color=color, linewidth=2, alpha=0.8, label='PCR(20日均线)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # 图表装饰
    plt.title('中金所300 PCR与认沽、认购成交量变化情况', fontsize=14, pad=15)
    fig.tight_layout()
    plt.figtext(0.85, 0.01, '数据来源：Tushare | 申万宏源研究', ha='center', fontsize=9)

    # 保存图表
    plt.savefig('图19_中金所300PCR与认沽认购成交量变化.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图19已保存")

    # 图20：中金所300波指与认沽、认购成交量之和趋势高度一致
    print("生成图20：中金所300波指与认沽、认购成交量之和趋势高度一致...")

    # 确保option_data包含VIX数据
    if 'vix' not in option_data.columns:
        # 获取VIX数据（假设有获取VIX数据的函数）
        vix_data = get_300_vix_data(start_date, end_date)
        # 转换vix_300的trade_date为datetime
        if 'trade_date' in vix_data.columns:
            vix_data['trade_date'] = pd.to_datetime(vix_data['trade_date'], errors='coerce')
            
        # 确保option_data的trade_date也是datetime
        if 'trade_date' in option_data.columns:
            option_data['trade_date'] = pd.to_datetime(option_data['trade_date'], errors='coerce')
        
        # 合并VIX数据到option_data
        option_data = pd.merge(
            option_data, 
            vix_data[['trade_date', 'vix']],
            on='trade_date',
            how='left'
        )
        
        print(f"已合并VIX数据，新增{len(vix_data)}条记录")

    # 创建20日均线列（带容错处理）
    if 'vix' in option_data.columns:
        # 数据清洗：转换数据类型
        option_data['vix'] = pd.to_numeric(option_data['vix'], errors='coerce')
        
        # 缺失值处理
        option_data['vix'].fillna(method='ffill', inplace=True)
        option_data['vix'].fillna(method='bfill', inplace=True)
        
        # 计算20日均线
        option_data['vix_ma20'] = option_data['vix'].rolling(
            window=20,
            min_periods=5  # 最小5个数据点
        ).mean()
        
        print("成功创建vix_ma20列")
    else:
        print("错误：option_data中缺少vix列")
        return  # 提前退出

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 成交量之和（左轴）
    color = 'tab:blue'
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('认沽+认购成交量(20日均线, 万手)', color=color, fontsize=12)
    ax1.plot(option_data['trade_date'], option_data['total_vol_ma20'], 
            color=color, linewidth=2, label='成交量之和')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # VIX波指（右轴）
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('300波指(20日均线)', color=color, fontsize=12)
    ax2.plot(option_data['trade_date'], option_data['vix_ma20'], 
            color=color, linewidth=2, alpha=0.8, label='300波指')
    ax2.tick_params(axis='y', labelcolor=color)

    # 图表装饰
    plt.title('中金所300波指与认沽、认购成交量之和趋势高度一致', fontsize=14, pad=15)
    fig.tight_layout()
    plt.figtext(0.85, 0.01, '数据来源：Tushare | 申万宏源研究', ha='center', fontsize=9)

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # 保存图表
    plt.savefig('图20_中金所300波指与成交量之和趋势.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图20已保存")

    # 图21：PCR和VIX与标的价格之间的相关性具有反向关系
    print("生成图21：PCR和VIX与标的价格之间的相关性具有反向关系...")

    # 准备数据 - 计算3个月滚动相关系数
    merged_data = pd.merge(option_data, hs300_data, on='trade_date', how='inner')
    merged_data['pcr_ma'] = merged_data['pcr_ma20'].rolling(window=20).mean()
    merged_data['vix_price_corr'] = merged_data['vix_ma20'].rolling(window=66).corr(merged_data['price_hs300'])

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 相关系数曲线
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('相关系数', color='k', fontsize=12)
    ax1.plot(merged_data['trade_date'], merged_data['vix_price_corr'], 
            color='tab:blue', linewidth=2, label='VIX与标的价格相关系数')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    # 沪深300指数（右轴）
    ax2 = ax1.twinx()
    ax2.plot(merged_data['trade_date'], merged_data['pcr_ma'], 
            color='tab:red', linewidth=2, label='PCR-ma')
    ax2.tick_params(axis='y', labelcolor=color)

    # 图表装饰
    plt.title('PCR和VIX与标的价格之间的相关性具有反向关系', fontsize=14, pad=15)
    fig.tight_layout()
    plt.figtext(0.85, 0.01, '数据来源：Tushare | 申万宏源研究 | 窗口:3个月(66个交易日)', ha='center', fontsize=9)

    # 保存图表
    plt.savefig('图21_PCR和VIX与标的价格相关性关系.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图21已保存")

    print("所有图表生成完成！")

if __name__ == '__main__':
    main()