import pandas as pd
import numpy as np
import tushare as ts
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 初始化Tushare
ts.set_token("4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77") 
pro = ts.pro_api()

def calculate_iv(S, K, T, r, option_price, option_type):
    """计算隐含波动率（Black-Scholes模型）"""
    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    try:
        # 使用布伦特方法求解
        return brentq(lambda sigma: bs_price(sigma) - option_price, 1e-6, 5.0)
    except:
        return np.nan

def get_vix_data(trade_date, underlying_price, rf_rate):
    """计算单日VIX指数"""
    # 获取所有有效期权合约
    df_basic = pro.opt_basic(exchange='SSE', fields='ts_code,call_put,exercise_price,list_date,delist_date')
    df_basic = df_basic[df_basic['list_date'] <= trade_date]
    df_basic = df_basic[df_basic['delist_date'] > trade_date]
    
    # 获取期权日行情
    df_daily = pro.opt_daily(trade_date=trade_date, exchange='SSE', fields='ts_code,close')
    
    # 合并数据
    df = pd.merge(df_basic, df_daily, on='ts_code')
    df['trade_date'] = trade_date
    df['underlying_price'] = underlying_price
    df['rf_rate'] = rf_rate
    
    # 计算到期时间
    trade_dt = datetime.strptime(trade_date, '%Y%m%d')
    df['delist_date_dt'] = pd.to_datetime(df['delist_date'])
    df['T'] = (df['delist_date_dt'] - trade_dt).dt.days / 365
    
    # 计算隐含波动率
    df['iv'] = df.apply(lambda row: calculate_iv(
        row['underlying_price'], 
        row['exercise_price'],
        row['T'],
        row['rf_rate'],
        row['close'],
        row['call_put']
    ), axis=1)
    
    # 筛选近月和次月合约
    expiries = df['delist_date'].unique()
    if len(expiries) < 2:
        return np.nan
    
    sorted_expiries = sorted(expiries)
    near_expiry = sorted_expiries[0]
    next_expiry = sorted_expiries[1]
    
    # 选取最接近平价的8个合约（4看涨+4看跌）
    def select_atm_options(df_expiry):
        df_expiry = df_expiry.copy()
        df_expiry['moneyness'] = abs(df_expiry['exercise_price'] - underlying_price)
        df_expiry = df_expiry.sort_values('moneyness')
        
        calls = df_expiry[df_expiry['call_put'] == 'call'].head(4)
        puts = df_expiry[df_expiry['call_put'] == 'put'].head(4)
        return pd.concat([calls, puts])
    
    near_options = select_atm_options(df[df['delist_date'] == near_expiry])
    next_options = select_atm_options(df[df['delist_date'] == next_expiry])
    
    if near_options.empty or next_options.empty:
        return np.nan
    
    # 计算时间加权VIX
    T1 = near_options['T'].mean()
    T2 = next_options['T'].mean()
    
    # 时间权重因子（30天为基准）
    w = (T2 - 30/365) / (T2 - T1)
    vix = w * near_options['iv'].mean() + (1 - w) * next_options['iv'].mean()
    
    return vix

def calculate_pcr(trade_date):
    """计算认沽认购比(PCR)"""
    # 获取所有期权合约
    df_daily = pro.opt_daily(trade_date=trade_date, exchange='SSE', fields='ts_code,call_put,vol')
    
    put_vol = df_daily[df_daily['call_put'] == 'put']['vol'].sum()
    call_vol = df_daily[df_daily['call_put'] == 'call']['vol'].sum()
    
    if call_vol > 0:
        return put_vol / call_vol
    return np.nan

def get_underlying_data(ts_code, start_date, end_date):
    """获取标的物价格数据"""
    if '510050' in ts_code:
        df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date, fields='trade_date,close')
    else:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date, fields='trade_date,close')
    return df.rename(columns={'close': 'price'})

def calculate_vix_series(ts_code, start_date, end_date):
    """计算VIX时间序列"""
    # 获取标的物价格
    underlying = get_underlying_data(ts_code, start_date, end_date)
    
    # 获取无风险利率（Shibor 1W）
    shibor = pro.shibor(start_date=start_date, end_date=end_date, fields='date,1w')
    shibor = shibor.rename(columns={'date': 'trade_date', '1w': 'rf_rate'})
    shibor['rf_rate'] = shibor['rf_rate'] / 100
    
    # 合并数据
    df = pd.merge(underlying, shibor, on='trade_date', how='left')
    
    # 计算每日VIX
    vix_values = []
    for i, row in df.iterrows():
        vix = get_vix_data(row['trade_date'], row['price'], row['rf_rate'])
        vix_values.append(vix)
    
    df['vix'] = vix_values
    df['vix_ma20'] = df['vix'].rolling(20).mean()
    
    return df

def plot_vix_comparison(vix_df, price_df, title):
    """绘制VIX与标的价格对比图"""
    merged = pd.merge(vix_df, price_df, on='trade_date', suffixes=('_vix', '_price'))
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(merged['trade_date'], merged['vix_ma20'], 'b-', label='VIX (20MA)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VIX', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(merged['trade_date'], merged['price'], 'r-', label='Underlying Price')
    ax2.set_ylabel('Price', color='r')
    ax2.tick_params('y', colors='r')
    
    plt.title(title)
    fig.tight_layout()
    plt.show()

def plot_correlation(vix_df, price_df, window, title):
    """绘制滚动相关系数图"""
    merged = pd.merge(vix_df, price_df, on='trade_date')
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged['rolling_corr'] = merged['vix'].rolling(window).corr(merged['price'])
    
    plt.figure(figsize=(14, 7))
    plt.plot(merged['trade_date'], merged['rolling_corr'], 'g-')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title(f'{title} ({window}D Rolling Correlation)')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.show()

def plot_pcr_analysis(ts_code, start_date, end_date):
    """绘制PCR相关分析图"""
    # 获取标的物价格
    price_df = get_underlying_data(ts_code, start_date, end_date)
    
    # 计算PCR
    pcr_values = []
    for date in price_df['trade_date']:
        pcr = calculate_pcr(date)
        pcr_values.append(pcr)
    
    price_df['pcr'] = pcr_values
    price_df['pcr_ma20'] = price_df['pcr'].rolling(20).mean()
    
    # 计算VIX
    vix_df = calculate_vix_series(ts_code, start_date, end_date)
    
    # 合并数据
    merged = pd.merge(price_df, vix_df, on='trade_date', suffixes=('', '_vix'))
    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    
    # 图18：PCR与价格
    plt.figure(figsize=(14, 7))
    plt.plot(merged['trade_date'], merged['pcr_ma20'], 'b-', label='PCR (20MA)')
    plt.legend(loc='upper left')
    plt.twinx()
    plt.plot(merged['trade_date'], merged['price'], 'r-', label='Price')
    plt.legend(loc='upper right')
    plt.title('PCR vs Underlying Price')
    plt.show()
    
    # 图21：PCR-VIX-价格相关性
    merged['rolling_corr_pcr'] = merged['pcr'].rolling(60).corr(merged['price'])
    merged['rolling_corr_vix'] = merged['vix'].rolling(60).corr(merged['price'])
    
    plt.figure(figsize=(14, 7))
    plt.plot(merged['trade_date'], merged['rolling_corr_pcr'], 'b-', label='PCR-Price Correlation')
    plt.plot(merged['trade_date'], merged['rolling_corr_vix'], 'r-', label='VIX-Price Correlation')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('PCR and VIX Correlation with Price (60D Rolling)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 主程序
def main():
    # 设置日期范围
    start_date = '20191223'
    end_date = '20241115'
    
    # 图14：50ETF波指与标的价格
    print("生成图14：50ETF波指与标的价格...")
    vix_50 = calculate_vix_series('510050.SH', start_date, end_date)
    price_50 = get_underlying_data('510050.SH', start_date, end_date)
    plot_vix_comparison(vix_50, price_50, '50ETF VIX vs Price')
    
    # 图15：300波指与标的价格
    print("生成图15：300波指与标的价格...")
    vix_300 = calculate_vix_series('000300.SH', start_date, end_date)
    price_300 = get_underlying_data('000300.SH', start_date, end_date)
    plot_vix_comparison(vix_300, price_300, 'CSI 300 VIX vs Price')
    
    # 图16：50ETF波指与标的价格3M滚动相关系数
    print("生成图16：50ETF波指与标的价格3M滚动相关系数...")
    plot_correlation(vix_50, price_50, 60, '50ETF VIX-Price Correlation')
    
    # 图17：300波指与标的价格3M滚动相关系数
    print("生成图17：300波指与标的价格3M滚动相关系数...")
    plot_correlation(vix_300, price_300, 60, 'CSI 300 VIX-Price Correlation')
    
    # 图18-21：PCR分析
    print("生成图18-21：PCR相关分析...")
    plot_pcr_analysis('000300.SH', start_date, end_date)

if __name__ == "__main__":
    main()