import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置部分 ==================== #
TOKEN = '4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77'
ts.set_token(TOKEN)
pro = ts.pro_api()

START_DATE = '20200101'
END_DATE = '20250718'
L_RET = 20  # 价格动量周期
L_VOL = 20  # 成交量周期
dual_neg_to_cash = True  # 启用双负过滤
neg_threshold = 0.00     # 双负阈值

# ==================== 数据获取 ==================== #
def get_index_data(ts_code, start_date, end_date):
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

# 获取上证50和中证500数据（包含开盘价）
large_df = get_index_data('000016.SH', START_DATE, END_DATE)
small_df = get_index_data('000852.SH', START_DATE, END_DATE)

# 合并数据
df = pd.DataFrame({
    'date': large_df['trade_date'],
    'large_close': large_df['close'].values,
    'large_open': large_df['open'].values,
    'large_vol': large_df['vol'].values
})
df = df.merge(small_df[['trade_date','close','open','vol']], 
              left_on='date', right_on='trade_date', how='inner')
df.rename(columns={'close':'small_close', 'open':'small_open', 'vol':'small_vol'}, inplace=True)
df.drop(columns='trade_date', inplace=True)
df.set_index('date', inplace=True)

# ==================== 策略实现 ==================== #
def calc_signals(base_df, L_ret, L_vol, dual_neg_to_cash, neg_threshold):
    df = base_df.copy()
    # 计算L_ret日收益率（基于收盘价）
    df['large_ret_L'] = df['large_close'] / df['large_close'].shift(L_ret) - 1
    df['small_ret_L'] = df['small_close'] / df['small_close'].shift(L_ret) - 1
    
    # 计算L_vol日平均成交量份额
    ma_large_vol = df['large_vol'].rolling(L_vol).mean()
    ma_small_vol = df['small_vol'].rolling(L_vol).mean()
    df['vs_large'] = ma_large_vol / (ma_large_vol + ma_small_vol)

    signal = []
    current = 0
    warmup = max(L_ret, L_vol)  # 预热期
    
    for i in range(len(df)):
        if i < warmup:
            signal.append(0)
            continue
            
        # 双负过滤条件
        if dual_neg_to_cash and \
           (df['large_ret_L'].iloc[i] < neg_threshold) and \
           (df['small_ret_L'].iloc[i] < neg_threshold):
            current = 0
        else:
            # 使用前日收盘价和当日开盘价综合判断
            # 计算当日开盘价相对前日收盘价的变动
            large_open_ret = df['large_open'].iloc[i] / df['large_close'].iloc[i-1] - 1
            small_open_ret = df['small_open'].iloc[i] / df['small_close'].iloc[i-1] - 1
            
            # 综合条件：结合前日收盘价动量与当日开盘强度
            cond_large = (df['large_ret_L'].iloc[i] > df['small_ret_L'].iloc[i]) and \
                         (df['vs_large'].iloc[i] > df['vs_large'].iloc[i-1]) 
                         
            cond_small = (df['small_ret_L'].iloc[i] > df['large_ret_L'].iloc[i]) and \
                         ((1 - df['vs_large'].iloc[i]) > (1 - df['vs_large'].iloc[i-1])) 

                         
            if cond_large:
                current = 1
            elif cond_small:
                current = -1
                
        signal.append(current)
    
    df['signal'] = signal
    return df

# 生成信号
signal_df = calc_signals(df, L_RET, L_VOL, dual_neg_to_cash, neg_threshold)

# ==================== 绩效计算 ==================== #
# 计算每日收益率（基于开盘价执行）
signal_df['ret_large'] = signal_df['large_open'].pct_change()
signal_df['ret_small'] = signal_df['small_open'].pct_change()

# 用当日信号在当日开盘执行（避免未来函数）
signal_df['strategy_ret'] = np.where(signal_df['signal'].shift(2) > 0, signal_df['ret_large'],
                                    np.where(signal_df['signal'].shift(2)  < 0, signal_df['ret_small'], 0))

# 加入手续费（千1.5），只在换仓时收取
fee_rate = 0.0001
switch_flag = (signal_df['signal'] != signal_df['signal'].shift(1))
signal_df['strategy_ret_net'] = signal_df['strategy_ret'] * (1 - switch_flag * fee_rate)

# 计算累计收益（含手续费）
signal_df['cum_strategy'] = (1 + signal_df['strategy_ret_net']).cumprod()
signal_df['cum_large'] = (1 + signal_df['ret_large']).cumprod()
signal_df['cum_small'] = (1 + signal_df['ret_small']).cumprod()

# ==================== 绩效分析 ==================== #
def calculate_performance(df, ret_col='strategy_ret_net'):
    total_return = df[ret_col].add(1).prod() - 1
    n_years = (df.index[-1] - df.index[0]).days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    cum_returns = df[ret_col].add(1).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    sharpe_ratio = df[ret_col].mean() / df[ret_col].std() * np.sqrt(252)

    gains = df[df[ret_col] > 0][ret_col]
    losses = df[df[ret_col] < 0][ret_col]
    profit_factor = gains.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf

    annual_bench = (1 + df['ret_large'].mean()) ** 252 - 1
    annual_excess = annual_return - annual_bench

    return {
        '总收益率': total_return,
        '年化收益率': annual_return,
        '年化超额收益率(相对大盘)': annual_excess,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '盈亏比': profit_factor
    }

performance = calculate_performance(signal_df)
print("\n改进策略绩效指标:")
for k, v in performance.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

position_count = signal_df['signal'].value_counts().sort_index()
position_map = {1: '持有大盘', -1: '持有小盘', 0: '空仓'}
print("\n各仓位出现次数统计：")
for k, v in position_count.items():
    print(f"{position_map.get(k, k)}: {v} 天")

# ==================== 可视化 ==================== #
plt.figure(figsize=(12, 6))
plt.plot(signal_df['cum_strategy'], label='改进策略净值', linewidth=2)
plt.plot(signal_df['cum_large'], label='上证50', alpha=0.7)
plt.plot(signal_df['cum_small'], label='中证500', alpha=0.7)

switch_dates = signal_df[signal_df['signal'].diff() != 0].index
for date in switch_dates:
    plt.axvline(date, color='gray', linestyle='--', alpha=0.3)

plt.title('改进版价量双动量策略 vs 大小盘指数表现 (2020-2025)')
plt.xlabel('日期')
plt.ylabel('累计收益率')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()