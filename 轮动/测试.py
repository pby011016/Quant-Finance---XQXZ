import tushare as ts
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 tushare token
ts.set_token('4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77')
pro = ts.pro_api()

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_or_fetch(name, fetch_func, force_reload=False):
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    if os.path.exists(path) and not force_reload:
        with open(path, 'rb') as f:
            return pickle.load(f)
    data = fetch_func()
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data

def get_index_daily(index_code, start_date, end_date):
    def fetch():
        return pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
    return load_or_fetch(f'{index_code}_daily_{start_date}_{end_date}', fetch)

def calc_rsi(close, window=60):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_psy(close, window):
    up_days = (close.diff() > 0).rolling(window).sum()
    return 100 * up_days / window

def preprocess_index_features(df, prefix):
    df = df.sort_values('trade_date')
    df['rsi_3m'] = calc_rsi(df['close'], 60)
    df['psy_6m'] = calc_psy(df['close'], 120)
    df['psy_12m'] = calc_psy(df['close'], 240)
    df = df[['trade_date', 'close', 'rsi_3m', 'psy_6m', 'psy_12m']].copy()
    df.columns = ['trade_date'] + [f'{prefix}_{col}' for col in df.columns[1:]]
    return df

def prepare_logit_data(start_date='20190101', end_date='20250630', holding_days=20):
    # 原始数据获取
    df_100 = get_index_daily('000016.SH', start_date, end_date)
    df_500 = get_index_daily('399006.SZ', start_date, end_date)

    # 技术指标计算
    df_100 = preprocess_index_features(df_100, '100')
    df_500 = preprocess_index_features(df_500, '500')
    
    # 合并时保留成交量数据
    df = pd.merge(
        df_100[['trade_date', '100_close', '100_rsi_3m', '100_psy_6m', '100_psy_12m']],
        df_500[['trade_date', '500_close', '500_rsi_3m', '500_psy_6m', '500_psy_12m']],
        on='trade_date'
    )
    
    # 添加原始成交量数据
    vol_100 = get_index_daily('000016.SH', start_date, end_date)[['trade_date', 'vol']]
    vol_500 = get_index_daily('399006.SZ', start_date, end_date)[['trade_date', 'vol']]
    df = pd.merge(df, vol_100.rename(columns={'vol':'100_vol'}), on='trade_date')
    df = pd.merge(df, vol_500.rename(columns={'vol':'500_vol'}), on='trade_date')
    
    # 计算特征差值
    df['rsi_diff'] = df['500_rsi_3m'] - df['100_rsi_3m']
    df['psy6_diff'] = df['500_psy_6m'] - df['100_psy_6m']
    df['psy12_diff'] = df['500_psy_12m'] - df['100_psy_12m']
    
    # 成交量特征（1个月比率）
    df['vol_ratio_1m'] = (df['500_vol'].rolling(20).mean() / 
                         df['100_vol'].rolling(20).mean())
    

    # 增加动量因子（如1个月、3个月收益率差）
    df['mom_1m_500'] = df['500_close'].pct_change(periods=20)
    df['mom_1m_100'] = df['100_close'].pct_change(periods=20)
    df['mom_1m_diff'] = df['mom_1m_500'] - df['mom_1m_100']

    df['mom_3m_500'] = df['500_close'].pct_change(periods=60)
    df['mom_3m_100'] = df['100_close'].pct_change(periods=60)
    df['mom_3m_diff'] = df['mom_3m_500'] - df['mom_3m_100']



    # 未来收益和标签
    df['ret_100'] = df['100_close'].pct_change().shift(-holding_days)
    df['ret_500'] = df['500_close'].pct_change().shift(-holding_days)
    df['target'] = (df['ret_500'] > df['ret_100']).astype(int)

    # 1. 获取CPI数据
    cpi = pro.cn_cpi(start_date=start_date, end_date=end_date)
    cpi = cpi.sort_values('month')
    cpi['month'] = pd.to_datetime(cpi['month'], format='%Y%m')
    cpi['cpi_yoy'] = pd.to_numeric(cpi['nt_yoy'], errors='coerce')  # 假设同比字段叫'yoy'

    # 2. 计算环比增速
    cpi['cpi_yoy_mom'] = cpi['cpi_yoy'].pct_change() * 100

    # 3. 滞后一期
    cpi['cpi_yoy_mom_lag1'] = cpi['cpi_yoy_mom'].shift(1)

    # 4. 合并到主表（trade_date转datetime，按月对齐）
    df['trade_month'] = pd.to_datetime(df['trade_date']).dt.to_period('M').dt.to_timestamp()
    cpi = cpi[['month', 'cpi_yoy_mom_lag1']].rename(columns={'month': 'trade_month'})
    df = pd.merge(df, cpi, on='trade_month', how='left')
    df = df.drop(columns=['trade_month'])

    # 1. 获取M2数据
    m2 = pro.cn_m(start_month=start_date[:6], end_month=end_date[:6])
    m2 = m2.sort_values('month')
    m2['month'] = pd.to_datetime(m2['month'], format='%Y%m')
    m2['m2_yoy'] = pd.to_numeric(m2['m2_yoy'], errors='coerce')

    # 2. 计算M2同比环比增速
    m2['m2_yoy_mom'] = m2['m2_yoy'].pct_change() * 100

    # 3. 滞后一期
    m2['m2_yoy_mom_lag1'] = m2['m2_yoy_mom'].shift(1)

    # 4. 合并到主表（trade_date转datetime，按月对齐）
    df['trade_month'] = pd.to_datetime(df['trade_date']).dt.to_period('M').dt.to_timestamp()
    m2 = m2[['month', 'm2_yoy_mom_lag1']].rename(columns={'month': 'trade_month'})
    df = pd.merge(df, m2, on='trade_month', how='left')
    df = df.drop(columns=['trade_month'])

    # ...已有代码...
    df = df.dropna().reset_index(drop=True)
    return df[['trade_date', 'rsi_diff', 'psy6_diff', 'psy12_diff', 
               'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1', 'm2_yoy_mom_lag1', 'target', 
               'ret_100', 'ret_500', '100_close', '500_close']]
    
def train_logit_model():
    df = prepare_logit_data()
    # X = df[['rsi_diff', 'psy6_diff', 'psy12_diff']]
    X = df[['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff']]
    y = df['target']
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'样本内准确率：{acc:.2%}')
    df['pred_prob'] = model.predict_proba(X)[:,1]
    return model, df

def max_drawdown(cum_returns):
    roll_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / roll_max - 1
    return drawdowns.min()

def backtest(df):
    df = df.copy()
    df['position'] = np.where(df['pred_prob'] > 0.5, 1, -1)
#     df['position'] = np.select(
#     [df['pred_prob'] > 0.55, df['pred_prob'] < 0.45],
#     [1, -1], 
#     default=0  # 40%-60%区间空仓
# )
    # df['spread_return'] = df['ret_500'] - df['ret_100']
    # df['strategy_ret'] = df['position'] * df['spread_return']
    # df['cum_ret'] = (1 + df['strategy_ret']).cumprod()
    # df['cum_500'] = (1 + df['ret_500']).cumprod()
    # df['cum_100'] = (1 + df['ret_100']).cumprod()

    # 轮动真实净值
    df['strategy_ret'] = np.where(df['position'] == 1, df['ret_500'], df['ret_100'])
    df['cum_ret'] = (1 + df['strategy_ret']).cumprod()
    df['cum_500'] = (1 + df['ret_500']).cumprod()
    df['cum_100'] = (1 + df['ret_100']).cumprod()
    # ...后续统计和画图不变...
    days = len(df)
    annual_ret = df['cum_ret'].iloc[-1] ** (252 / days) - 1
    total_ret = df['cum_ret'].iloc[-1] - 1
    annual_vol = df['strategy_ret'].std() * np.sqrt(252)
    sharpe = (annual_ret - 0.02) / annual_vol

    win_ratio = (df['strategy_ret'] > 0).mean()
    pnl_ratio = df[df['strategy_ret'] > 0]['strategy_ret'].mean() / abs(df[df['strategy_ret'] < 0]['strategy_ret'].mean())

    excess_annual = annual_ret - (df['cum_100'].iloc[-1] ** (252 / days) - 1)
    max_dd = max_drawdown(df['cum_ret'])

    print(f'总收益率: {total_ret:.2%}')
    print(f'年化收益率: {annual_ret:.2%}')
    print(f'年化超额收益率: {excess_annual:.2%}')
    print(f'最大回撤: {max_dd:.2%}')
    print(f'夏普比率: {sharpe:.2f}')
    print(f'盈亏比: {pnl_ratio:.2f}')
    print(f'胜率: {win_ratio:.2%}')

    df['date'] = pd.to_datetime(df['trade_date'])
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['cum_ret'], label='策略净值')
    plt.plot(df['date'], df['cum_500'], label='中证500')
    plt.plot(df['date'], df['cum_100'], label='中证100')
    plt.legend()
    plt.title('策略净值 vs 大小盘指数')
    plt.xlabel('日期')
    plt.ylabel('累计收益')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # ...已有统计和画图...

    # 统计轮动次数
    switch_count = (df['position'] != df['position'].shift(1)).sum()
    print(f'轮动次数: {switch_count}')

    return df[['trade_date', 'strategy_ret', 'cum_ret', 'position']]
    # return df[['trade_date', 'strategy_ret', 'cum_ret', 'position']]


def rolling_walk_forward(df, features, window_train=50, window_test=1):
    results = []
    n = len(df)
    for start in range(0, n - window_train - window_test + 1, window_test):
        train = df.iloc[start : start + window_train].copy()
        test = df.iloc[start + window_train : start + window_train + window_test].copy()
        train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['target'])
        test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['target'])
        if train.empty or test.empty:
            continue
        X_train, y_train = train[features], train['target']
        X_test, y_test = test[features], test['target']
        model = LogisticRegression()
        model.fit(X_train, y_train)
        test = test.copy()
        test['pred_prob'] = model.predict_proba(X_test)[:, 1]
        # 只做多修改点1：仓位映射为0或1
        test['position'] = np.where(test['pred_prob'] > 0.5, 1, 0)
        # 只做多修改点2：收益计算仅持有多头
        test['strategy_ret'] = np.where(
            test['position'] == 1, 
            test['ret_500'], 
            test['ret_100']  # 或0若完全空仓
        )
        test['cum_ret'] = (1 + test['strategy_ret']).cumprod()
        test['window_id'] = start // window_test + 1
        results.append(test)
    return pd.concat(results) if results else pd.DataFrame()


# def grid_search_rolling_window(df, features, train_range, test_range):
#     """
#     遍历不同训练集和测试集长度，寻找最优组合
#     train_range: 训练集长度列表，如 [400, 500, 600, 700]
#     test_range: 测试集长度列表，如 [20, 60, 100]
#     """
#     results = []
#     for window_train in train_range:
#         for window_test in test_range:
#             df_oos = rolling_walk_forward(df, features, window_train, window_test)
#             if df_oos.empty:
#                 continue
#             days = len(df_oos)
#             if days == 0:
#                 continue
#             annual_ret = (df_oos['cum_ret'].iloc[-1]) ** (252 / days) - 1
#             annual_vol = df_oos['strategy_ret'].std() * np.sqrt(252)
#             sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else np.nan
#             max_dd = max_drawdown(df_oos['cum_ret'])
#             results.append({
#                 'window_train': window_train,
#                 'window_test': window_test,
#                 'annual_ret': annual_ret,
#                 'sharpe': sharpe,
#                 'max_dd': max_dd
#             })
#             print(f"train={window_train}, test={window_test}, 年化收益={annual_ret:.2%}, 夏普={sharpe:.2f}, 最大回撤={max_dd:.2%}")
#     return pd.DataFrame(results)

# # 主函数调用示例
# if __name__ == '__main__':
#     df = prepare_logit_data()
#     features = ['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff']
#     train_range = list(range(200, 1001, 30))
#     test_range = list(range(100, 501, 20))
#     result_df = grid_search_rolling_window(df, features, train_range, test_range)
#     print(result_df.sort_values('annual_ret', ascending=False))

# ====== 修改主函数调用 ======
if __name__ == '__main__':
    df = prepare_logit_data()
    # features = ['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff']
    # features = ['rsi_diff', 'psy6_diff', 'psy12_diff', 'vol_ratio_1m']
    df .to_excel
    print(df.head())
    features = ['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1', 'm2_yoy_mom_lag1']
    df_oos = rolling_walk_forward(df, features, window_train=500, window_test=100)
    print("样本外窗口合并后数据：")
    print(df_oos.head())
    # 样本外整体回测
    backtest(df_oos)