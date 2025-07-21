# -*- coding: utf-8 -*-
"""
模块 1-5：数据下载 + 指标计算 + Logit模型训练 + 回测绩效评估 + 净值曲线绘图 + 收益率指标增强
"""

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

# def prepare_logit_data(start_date='20070201', end_date='20111231'):
#     df_100 = get_index_daily('000903.SH', start_date, end_date)
#     df_500 = get_index_daily('000905.SH', start_date, end_date)
#     df_100 = preprocess_index_features(df_100, '100')
#     df_500 = preprocess_index_features(df_500, '500')

#     df = pd.merge(df_100, df_500, on='trade_date')
#     df['rsi_diff'] = df['500_rsi_3m'] - df['100_rsi_3m']
#     df['psy6_diff'] = df['500_psy_6m'] - df['100_psy_6m']
#     df['psy12_diff'] = df['500_psy_12m'] - df['100_psy_12m']
#     df['ret_100'] = df['100_close'].pct_change().shift(-20)
#     df['ret_500'] = df['500_close'].pct_change().shift(-20)
#     df['target'] = (df['ret_500'] > df['ret_100']).astype(int)
#     df = df.dropna().reset_index(drop=True)
#     return df[['trade_date', 'rsi_diff', 'psy6_diff', 'psy12_diff', 'target', 'ret_100', 'ret_500', '100_close', '500_close']]


def prepare_logit_data(start_date='20070201', end_date='20111231', holding_days=20):
    # 原始数据获取（增加成交量）
    df_100 = get_index_daily('000903.SH', start_date, end_date)
    df_500 = get_index_daily('000905.SH', start_date, end_date)
    
    # 技术指标计算（保留原有逻辑）
    df_100 = preprocess_index_features(df_100, '100')
    df_500 = preprocess_index_features(df_500, '500')
    
    # 合并时保留成交量数据
    df = pd.merge(
        df_100[['trade_date', '100_close', '100_rsi_3m', '100_psy_6m', '100_psy_12m']],
        df_500[['trade_date', '500_close', '500_rsi_3m', '500_psy_6m', '500_psy_12m']],
        on='trade_date'
    )
    
    # 添加原始成交量数据（需重新获取）
    vol_100 = get_index_daily('000903.SH', start_date, end_date)[['trade_date', 'vol']]
    vol_500 = get_index_daily('000905.SH', start_date, end_date)[['trade_date', 'vol']]
    df = pd.merge(df, vol_100.rename(columns={'vol':'100_vol'}), on='trade_date')
    df = pd.merge(df, vol_500.rename(columns={'vol':'500_vol'}), on='trade_date')
    
    # 计算特征差值（原有）
    df['rsi_diff'] = df['500_rsi_3m'] - df['100_rsi_3m']
    df['psy6_diff'] = df['500_psy_6m'] - df['100_psy_6m']
    df['psy12_diff'] = df['500_psy_12m'] - df['100_psy_12m']
    
    # 新增成交量特征（1个月比率）
    df['vol_ratio_1m'] = (df['500_vol'].rolling(20).mean() / 
                         df['100_vol'].rolling(20).mean())
    
    # 未来收益和标签（原有）
    df['ret_100'] = df['100_close'].pct_change().shift(-holding_days)
    df['ret_500'] = df['500_close'].pct_change().shift(-holding_days)
    df['target'] = (df['ret_500'] > df['ret_100']).astype(int)
    
    # 清理数据
    df = df.dropna().reset_index(drop=True)
    
    return df[['trade_date', 'rsi_diff', 'psy6_diff', 'psy12_diff', 
              'vol_ratio_1m', 'target', 'ret_100', 'ret_500',
              '100_close', '500_close']]

def train_logit_model():
    df = prepare_logit_data()
    X = df[['rsi_diff', 'psy6_diff', 'psy12_diff']]
    y = df['target']
    model = LogisticRegression()
    model.fit(X, y)
    # # 新增校准代码
    # from sklearn.calibration import CalibratedClassifierCV
    # calibrated_model = CalibratedClassifierCV(
    #     model, 
    #     cv=5, 
    #     method='isotonic'
    # )
    # calibrated_model.fit(X, y)
    # df['calibrated_prob'] = calibrated_model.predict_proba(X)[:,1]
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
#     [df['pred_prob'] > 0.65, df['pred_prob'] < 0.35],
#     [1, -1], 
#     default=0  # 35%-65%区间空仓
# )
    df['spread_return'] = df['ret_500'] - df['ret_100']
    # # 在df['strategy_ret']计算前添加
    # transaction_cost = 0.001  # 单边手续费0.1%
    # df['turnover'] = df['position'].diff().abs()  # 换手标记
    # df['strategy_ret'] = (df['position'] * df['spread_return'] - 
    #                     df['turnover'] * transaction_cost)
    df['strategy_ret'] = df['position'] * df['spread_return']
    df['cum_ret'] = (1 + df['strategy_ret']).cumprod()
    df['cum_500'] = (1 + df['ret_500']).cumprod()
    df['cum_100'] = (1 + df['ret_100']).cumprod()

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

    return df[['trade_date', 'strategy_ret', 'cum_ret', 'position']]

# def oos_test_report():
#     """ 严格按研报第14页格式的测试 """
#     # 获取训练集
#     train = prepare_logit_data(*TIME_CONFIG['train'])
    
#     # 模型训练（需新增特征参数）
#     features = ['rsi_diff', 'psy6_diff', 'psy12_diff', 'vol_ratio_1m']
#     model = LogisticRegression().fit(train[features], train['target'])
    
#     # 按月滚动测试
#     dates = pd.date_range(*map(pd.to_datetime, TIME_CONFIG['test']), freq='MS')
#     results = []
#     for i in range(len(dates)-1):
#         test = prepare_logit_data(
#             start_date=dates[i].strftime('%Y%m%d'),
#             end_date=dates[i+1].strftime('%Y%m%d')
#         )
#         test['position'] = model.predict(test[features])
#         results.append(test)
    
#     return pd.concat(results)

def oos_test_report(start_date='20110101', end_date='20120831'):
    """ 样本外测试函数 """
    # 获取训练集（2007-2010）
    train = prepare_logit_data('20070201', '20101231')  
    
    # 获取测试集（2011-2012）
    test = prepare_logit_data(start_date, end_date)
    
    # 特征选择（包含新增的成交量因子）
    features = ['rsi_diff', 'psy6_diff', 'psy12_diff', 'vol_ratio_1m']
    
    # 模型训练与预测
    model = LogisticRegression()
    model.fit(train[features], train['target'])
    test['pred_prob'] = model.predict_proba(test[features])[:,1]
    
    # 回测
    return test

# def plot_report_results(df):
#     """ 生成研报第14页风格的净值曲线 """
#     plt.figure(figsize=(12,6))
    
#     # 区域填充
#     plt.fill_between(df['date'], df['cum_ret'], 
#                     where=(df['position']==1),
#                     color='r', alpha=0.2, label='小盘占优')
#     plt.fill_between(df['date'], df['cum_ret'],
#                     where=(df['position']==-1),
#                     color='g', alpha=0.2, label='大盘占优')
    
#     # 净值曲线
#     plt.plot(df['date'], df['cum_ret'], 'k-', lw=1.5, label='策略净值')
#     plt.title('大小盘轮动策略样本外表现(2010-2012)')
#     plt.legend(loc='upper left')

# 新增时间配置（紧邻主程序）
TIME_CONFIG = {
    'train': ('20070201', '20091231'),  # 训练集（研报使用前3年）
    'test': ('20100101', '20120831')    # 测试集（研报样本外区间）
}

if __name__ == '__main__':
    model, df_logit = train_logit_model()
    df_bt = backtest(df_logit)
    print(df_bt.tail())

if __name__ == '__main__':
    # 1. 训练阶段（使用2007-2009数据）
    train_data = prepare_logit_data(*TIME_CONFIG['train'])
    features = ['rsi_diff', 'psy6_diff', 'psy12_diff', 'vol_ratio_1m']
    model = LogisticRegression().fit(train_data[features], train_data['target'])
    
    # 2. 样本外测试（2010-2012）
    test_data = oos_test_report()
    test_results = backtest(test_data)  # 复用原有回测函数

    # 3. 研报式输出
    # plot_report_results(test_results)
    print(f"年化收益: {(test_results['cum_ret'].iloc[-1]**(12/32)-1):.1%}")  # 32个月

# if __name__ == '__main__':
#     # ===== 配置区 =====
#     FEATURE_COLS = ['rsi_diff', 'psy6_diff', 'psy12_diff', 'vol_ratio_1m']
#     TARGET_COL = 'target'
    
#     # ===== 训练阶段 =====
#     # 获取训练数据（2007-2009）
#     train_data = prepare_logit_data(*TIME_CONFIG['train'])
    
#     # 模型训练
#     model = LogisticRegression()
#     model.fit(train_data[FEATURE_COLS], train_data[TARGET_COL])
    
#     # ===== 测试阶段 =====
#     # 获取测试数据（2010-2012）
#     test_data = prepare_logit_data(*TIME_CONFIG['test'])
    
#     # 生成预测概率
#     test_data['pred_prob'] = model.predict_proba(test_data[FEATURE_COLS])[:,1]
    
#     # 执行回测
#     results = backtest(test_data)
    
#     # 可视化
#     plot_report_results(results)