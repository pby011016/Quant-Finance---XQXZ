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
neg_threshold = -0.01      # 双负阈值

# ==================== 数据获取 ==================== #
def get_index_data(ts_code, start_date, end_date):
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df

# 获取上证50和中证500数据
large_df = get_index_data('000016.SH', START_DATE, END_DATE)
small_df = get_index_data('000852.SH', START_DATE, END_DATE)

# 合并数据
df = pd.DataFrame({
    'date': large_df['trade_date'],
    'large_close': large_df['close'].values,
    'large_vol': large_df['vol'].values
})
df = df.merge(small_df[['trade_date','close','vol']], 
              left_on='date', right_on='trade_date', how='inner')
df.rename(columns={'close':'small_close','vol':'small_vol'}, inplace=True)
df.drop(columns='trade_date', inplace=True)
df.set_index('date', inplace=True)

# ==================== 策略实现 ==================== #
def calc_signals(base_df, L_ret, L_vol, dual_neg_to_cash, neg_threshold):
    df = base_df.copy()
    # 计算L_ret日收益率
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
            # 大盘条件：价格动量强且成交量份额上升
            cond_large = (df['large_ret_L'].iloc[i] > df['small_ret_L'].iloc[i]) and \
                         (df['large_ret_L'].iloc[i-1] > df['small_ret_L'].iloc[i-1]) and \
                         (df['vs_large'].iloc[i] > df['vs_large'].iloc[i-1])
            
            # 小盘条件：价格动量强且大盘成交量份额下降(即小盘份额上升)
            cond_small = (df['small_ret_L'].iloc[i] > df['large_ret_L'].iloc[i]) and \
                         (df['small_ret_L'].iloc[i-1] > df['large_ret_L'].iloc[i-1]) and \
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
# 计算每日收益率
signal_df['ret_large'] = signal_df['large_close'].pct_change()
signal_df['ret_small'] = signal_df['small_close'].pct_change()
# 用shift(1)将信号整体向后移动一天，避免未来函数
signal_df['signal_shift'] = signal_df['signal'].shift(1).fillna(0)
signal_df['strategy_ret'] = np.where(signal_df['signal_shift'] > 0, signal_df['ret_large'],
                                     np.where(signal_df['signal_shift'] < 0, signal_df['ret_small'], 0))

# 加入手续费（千1.5），只在换仓时收取
fee_rate = 0.0015
signal_df['signal_shift'] = signal_df['signal'].shift(1).fillna(0)
# 换仓点（前后signal不同且都不为0，或从0到非0/非0到0都收一次手续费）
switch_flag = (signal_df['signal'] != signal_df['signal_shift'])
# 只在换仓日扣除手续费
signal_df['strategy_ret_net'] = signal_df['strategy_ret'] *(1- switch_flag * fee_rate)

# 计算累计收益（含手续费）
signal_df['cum_strategy'] = (1 + signal_df['strategy_ret_net']).cumprod()
signal_df['cum_large'] = (1 + signal_df['ret_large']).cumprod()
signal_df['cum_small'] = (1 + signal_df['ret_small']).cumprod()

# 后续绩效统计、画图等都用 strategy_ret_net 和 cum_strategy

# 绩效指标计算
def calculate_performance(df, ret_col='strategy_ret'):
    total_return = df[ret_col].add(1).prod() - 1
    # 用实际年数计算年化收益率
    n_years = (df.index[-1] - df.index[0]).days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1

    # 计算最大回撤
    cum_returns = df[ret_col].add(1).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    # 计算夏普比率(假设无风险利率为0)
    sharpe_ratio = df[ret_col].mean() / df[ret_col].std() * np.sqrt(252)

    # 计算盈亏比
    gains = df[df[ret_col] > 0][ret_col]
    losses = df[df[ret_col] < 0][ret_col]
    profit_factor = gains.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf

    # 年化超额收益率
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
print("\n原始策略绩效指标:")
for k, v in performance.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# 统计各个时间的仓位状况
position_count = signal_df['signal'].value_counts().sort_index()
position_map = {1: '持有大盘', -1: '持有小盘', 0: '空仓'}
print("\n各仓位出现次数统计：")
for k, v in position_count.items():
    print(f"{position_map.get(k, k)}: {v} 天")

print("\n各仓位占比：")
total_days = len(signal_df)
for k, v in position_count.items():
    print(f"{position_map.get(k, k)}: {v/total_days:.2%}")

# ==================== 可视化 ==================== #
plt.figure(figsize=(12, 6))
plt.plot(signal_df['cum_strategy'], label='策略净值', linewidth=2)
plt.plot(signal_df['cum_large'], label='上证50', alpha=0.7)
plt.plot(signal_df['cum_small'], label='中证500', alpha=0.7)

# 标记信号切换点
switch_dates = signal_df[signal_df['signal'].diff() != 0].index
for date in switch_dates:
    plt.axvline(date, color='gray', linestyle='--', alpha=0.3)

plt.title('原始价量双动量策略 vs 大小盘指数表现 (2020-2024)')
plt.xlabel('日期')
plt.ylabel('累计收益率')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()