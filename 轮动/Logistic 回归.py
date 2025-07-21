import tushare as ts
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import warnings
warnings.filterwarnings('ignore')

# GPU支持检测
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CUDA GPU 可用，将使用GPU加速")
except ImportError:
    GPU_AVAILABLE = False
    print("CUDA GPU 不可用，将使用CPU多线程")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 初始化Tushare
try:
    pro = ts.pro_api('4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77')  
    print("Tushare初始化成功")
except Exception as e:
    print(f"Tushare初始化失败: {e}")
    raise


# # 设置 tushare token
# ts.set_token('4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77')
# pro = ts.pro_api('4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77')

# 设置输出目录
OUTPUT_DIR = os.path.join('result', 'logistic 回归')
CACHE_DIR = os.path.join('cache', 'logistic')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局数据缓存
GLOBAL_DATA_CACHE = {}

def get_cached_data(holding_days):
    """
    获取缓存的数据，如果不存在则创建
    """
    if holding_days not in GLOBAL_DATA_CACHE:
        print(f"首次加载holding_days={holding_days}的数据...")
        df = prepare_logit_data(holding_days=holding_days)
        features = ['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1','m2_yoy_mom_lag1']
        
        # 数据质量检查
        df_clean = df[features].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        X = df_clean.values
        y = (df['ret_500'] > df['ret_100']).astype(int).values
        
        scaler = StandardScaler()
        X_std_full = scaler.fit_transform(X)
        
        GLOBAL_DATA_CACHE[holding_days] = {
            'df': df,
            'X_std_full': X_std_full,
            'y': y,
            'dates': df['trade_date'].values
        }
        print(f"holding_days={holding_days}的数据已缓存")
    
    return GLOBAL_DATA_CACHE[holding_days]

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

def prepare_logit_data(start_date='20180101', end_date='20250630', holding_days=20):
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
    cpi['cpi_yoy_mom'] = cpi['cpi_yoy'].pct_change(fill_method=None) * 100

    # 3. 滞后一期
    cpi['cpi_yoy_mom_lag1'] = cpi['cpi_yoy_mom'].shift(1)

    # 4. 处理CPI数据的滞后性：根据交易日期匹配对应的CPI数据
    # 当月15日之前使用上上个月的数据，15日之后使用上个月的数据
    df['trade_date_dt'] = pd.to_datetime(df['trade_date'])
    
    # 创建CPI数据映射
    cpi_mapping = {}
    for _, row in cpi.iterrows():
        month = row['month']
        cpi_value = row['cpi_yoy_mom_lag1']
        
        # 为每个月的1-14日和15日之后分别映射
        # 1-14日使用上上个月的数据
        early_month = month - pd.DateOffset(months=2)
        # 15日之后使用上个月的数据  
        late_month = month - pd.DateOffset(months=1)
        
        cpi_mapping[(early_month.year, early_month.month, 'early')] = cpi_value
        cpi_mapping[(late_month.year, late_month.month, 'late')] = cpi_value
    
    # 根据交易日期匹配CPI数据
    df['cpi_yoy_mom_lag1'] = df.apply(
        lambda row: cpi_mapping.get(
            (row['trade_date_dt'].year, row['trade_date_dt'].month, 
             'early' if row['trade_date_dt'].day <= 14 else 'late'), 
            np.nan
        ), axis=1
    )

    # 1. 获取M2数据
    m2 = pro.cn_m(start_month=start_date[:6], end_month=end_date[:6])
    m2 = m2.sort_values('month')
    m2['month'] = pd.to_datetime(m2['month'], format='%Y%m')
    m2['m2_yoy'] = pd.to_numeric(m2['m2_yoy'], errors='coerce')

    # 2. 计算M2同比环比增速
    m2['m2_yoy_mom'] = m2['m2_yoy'].pct_change(fill_method=None) * 100

    # 3. 滞后一期
    m2['m2_yoy_mom_lag1'] = m2['m2_yoy_mom'].shift(1)

    # 4. 处理M2数据的滞后性：根据交易日期匹配对应的M2数据
    # 当月15日之前使用上上个月的数据，15日之后使用上个月的数据
    m2_mapping = {}
    for _, row in m2.iterrows():
        month = row['month']
        m2_value = row['m2_yoy_mom_lag1']
        
        # 为每个月的1-14日和15日之后分别映射
        # 1-14日使用上上个月的数据
        early_month = month - pd.DateOffset(months=2)
        # 15日之后使用上个月的数据  
        late_month = month - pd.DateOffset(months=1)
        
        m2_mapping[(early_month.year, early_month.month, 'early')] = m2_value
        m2_mapping[(late_month.year, late_month.month, 'late')] = m2_value
    
    # 根据交易日期匹配M2数据
    df['m2_yoy_mom_lag1'] = df.apply(
        lambda row: m2_mapping.get(
            (row['trade_date_dt'].year, row['trade_date_dt'].month, 
             'early' if row['trade_date_dt'].day <= 14 else 'late'), 
            np.nan
        ), axis=1
    )
    
    # 清理临时列
    df = df.drop(columns=['trade_date_dt'])

    # ...已有代码...
    df = df.dropna().reset_index(drop=True)
    
    # 数据清理：处理无穷大值
    numeric_columns = ['rsi_diff', 'psy6_diff', 'psy12_diff', 'vol_ratio_1m', 
                      'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1', 'm2_yoy_mom_lag1']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    return df[['trade_date', 'rsi_diff', 'psy6_diff', 'psy12_diff', 
               'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1', 'm2_yoy_mom_lag1', 'target', 
               'ret_100', 'ret_500', '100_close', '500_close']]

def train_logit_model():
    df = prepare_logit_data()
    # X = df[['rsi_diff', 'psy6_diff', 'psy12_diff']]
    X = df[['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1','m2_yoy_mom_lag1']]
    y = df['target']
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'样本内准确率：{acc:.2%}')
    # 添加clip限制，确保概率值在合理范围内
    df['pred_prob'] = np.clip(model.predict_proba(X)[:,1], 0.001, 0.999)
    return model, df

def max_drawdown(cum_returns):
    roll_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / roll_max - 1
    return drawdowns.min()

def backtest(df):
    df = df.copy()
    
    # 计算累计收益
    df['cum_ret'] = (1 + df['strategy_ret']).cumprod()
    
    # 计算统计指标
    days = len(df)
    annual_ret = df['cum_ret'].iloc[-1] ** (252 / days) - 1
    total_ret = df['cum_ret'].iloc[-1] - 1
    annual_vol = df['strategy_ret'].std() * np.sqrt(252)
    sharpe = (annual_ret - 0.02) / annual_vol

    win_ratio = (df['strategy_ret'] > 0).mean()
    pnl_ratio = df[df['strategy_ret'] > 0]['strategy_ret'].mean() / abs(df[df['strategy_ret'] < 0]['strategy_ret'].mean())

    max_dd = max_drawdown(df['cum_ret'])

    print(f'总收益率: {total_ret:.2%}')
    print(f'年化收益率: {annual_ret:.2%}')
    print(f'最大回撤: {max_dd:.2%}')
    print(f'夏普比率: {sharpe:.2f}')
    print(f'盈亏比: {pnl_ratio:.2f}')
    print(f'胜率: {win_ratio:.2%}')

    # 可视化
    df['date'] = pd.to_datetime(df['trade_date'])
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['cum_ret'], label='策略累积收益')
    plt.xlabel('日期')
    plt.ylabel('累积收益')
    plt.title('滚动窗口回测结果')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图片到输出目录
    plot_path = os.path.join(OUTPUT_DIR, 'backtest_result.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"回测结果图片已保存到: {plot_path}")

    # 统计轮动次数
    switch_count = (df['position'] != df['position'].shift(1)).sum()
    print(f'轮动次数: {switch_count}')

    # 保存回测结果数据
    backtest_data_path = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
    df[['trade_date', 'strategy_ret', 'cum_ret', 'position']].to_csv(backtest_data_path, index=False)
    print(f"回测结果数据已保存到: {backtest_data_path}")

    return df[['trade_date', 'strategy_ret', 'cum_ret', 'position']]

def train_logistic_regression_detailed(X_train, y_train, max_iter=1000, tol=1e-4):
    """
    详细的逻辑回归训练过程，展示参数迭代
    """
    n_samples, n_features = X_train.shape
    
    # 初始化参数
    beta = np.zeros(n_features + 1)  # +1 for intercept
    X_with_intercept = np.column_stack([np.ones(n_samples), X_train])
    
    print(f"开始训练逻辑回归...")
    print(f"样本数: {n_samples}, 特征数: {n_features}")
    print(f"最大迭代次数: {max_iter}")
    
    for iteration in range(max_iter):
        # 1. 计算当前预测概率
        z = X_with_intercept @ beta
        p_pred = 1 / (1 + np.exp(-z))
        
        # 2. 计算对数似然损失
        p_pred = np.clip(p_pred, 1e-15, 1-1e-15)
        log_likelihood = -np.sum(y_train * np.log(p_pred) + (1 - y_train) * np.log(1 - p_pred))
        
        # 3. 计算梯度
        gradient = X_with_intercept.T @ (y_train - p_pred)
        
        # 4. 计算Hessian矩阵
        W = np.diag(p_pred * (1 - p_pred))
        hessian = X_with_intercept.T @ W @ X_with_intercept
        
        # 5. 牛顿-拉夫森更新
        try:
            beta_new = beta - np.linalg.inv(hessian) @ gradient
        except np.linalg.LinAlgError:
            # 如果Hessian不可逆，使用伪逆
            beta_new = beta - np.linalg.pinv(hessian) @ gradient
        
        # 6. 检查收敛
        beta_change = np.linalg.norm(beta_new - beta)
        
        if iteration % 100 == 0:
            print(f"迭代 {iteration}: 损失={log_likelihood:.4f}, 参数变化={beta_change:.6f}")
        
        if beta_change < tol:
            print(f"收敛于迭代 {iteration}")
            break
            
        beta = beta_new
    
    print(f"最终损失: {log_likelihood:.4f}")
    print(f"最终参数: {beta}")
    
    return beta

def demonstrate_likelihood_vs_loss(X_train, y_train, model):
    """
    展示对数似然函数和损失函数的关系
    """
    # 获取预测概率
    y_pred_prob = model.predict_proba(X_train)[:, 1]
    
    # 计算对数似然
    y_pred_prob = np.clip(y_pred_prob, 1e-15, 1-1e-15)  # 避免log(0)
    log_likelihood = np.sum(y_train * np.log(y_pred_prob) + (1 - y_train) * np.log(1 - y_pred_prob))
    
    # 计算损失函数
    loss = -log_likelihood
    
    print(f"对数似然函数值: {log_likelihood:.4f}")
    print(f"损失函数值: {loss:.4f}")
    print(f"关系验证: 损失函数 = -对数似然函数: {loss:.4f} = -({log_likelihood:.4f})")
    
    return log_likelihood, loss

def rolling_walk_forward(df, features, window_train=300):
    """
    滚动窗口预测：用历史数据训练，对下一天预测
    window_train: 训练集长度（历史数据天数）
    """
    # 准备数据
    df = df.sort_values('trade_date')
    
    # 数据清理：处理无穷大和NaN值
    df_clean = df[features].copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    X = df_clean.values
    y = (df['ret_500'] > df['ret_100']).astype(int).values
    dates = df['trade_date'].values
    
    # 标准化整个样本
    scaler = StandardScaler()
    X_std_full = scaler.fit_transform(X)
    
    # 滚动窗口预测
    pred_dates = []
    pred_probs = []
    signals = []
    strat_rets = []
    
    for i in range(window_train, len(X_std_full)):
        # 训练集：从第1天到第i-1天（共window_train天）
        start_train = i - window_train
        end_train = i
        
        X_train = X_std_full[start_train:end_train]
        y_train = y[start_train:end_train]
        
        # 训练模型
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # 展示第一次训练的对数似然和损失函数关系
        if i == window_train:
            print(f"\n=== 第{i}天训练的对数似然和损失函数关系 ===")
            demonstrate_likelihood_vs_loss(X_train, y_train, model)
        
        # 对第 i 条样本预测
        p_i = model.predict_proba(X_std_full[i].reshape(1, -1))[0, 1]
        sig = 1 if p_i > 0.5 else -1

        
        # # 策略收益：多小盘空大盘
        # strat_ret = sig * (df.iloc[i]['ret_500'] - df.iloc[i]['ret_100'])
        strat_ret = df.iloc[i]['ret_500'] if sig == 1 else df.iloc[i]['ret_100']
        
        # 保存
        pred_dates.append(dates[i])
        pred_probs.append(p_i)
        signals.append(sig)
        strat_rets.append(strat_ret)
    
    # 汇总结果
    result = pd.DataFrame({
        'trade_date': pred_dates,
        'pred_prob': pred_probs,
        'position': signals,
        'strategy_ret': strat_rets
    })
    
    return result

def evaluate_parameter_combination(params):
    """
    评估单个参数组合的性能
    """
    holding_days, window_train = params
    
    try:
        # 直接调用prepare_logit_data获取数据，避免依赖全局缓存
        df = prepare_logit_data(holding_days=holding_days)
        features = ['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1','m2_yoy_mom_lag1']
        
        # 数据质量检查
        df_clean = df[features].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        X = df_clean.values
        y = (df['ret_500'] > df['ret_100']).astype(int).values
        
        # 标准化
        scaler = StandardScaler()
        X_std_full = scaler.fit_transform(X)
        
        if len(df) <= window_train:
            return params, -999, -999, -999, -999, -999, -999
        
        # 滚动窗口预测
        pred_dates = []
        strat_rets = []
        
        for i in range(window_train, len(X_std_full)):
            start_train = i - window_train
            end_train = i
            
            X_train = X_std_full[start_train:end_train]
            y_train = y[start_train:end_train]
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            p_i = model.predict_proba(X_std_full[i].reshape(1, -1))[0, 1]
            sig = 1 if p_i > 0.5 else -1
            # strat_ret = sig * (df.iloc[i]['ret_500'] - df.iloc[i]['ret_100'])
            strat_ret = df.iloc[i]['ret_500'] if sig == 1 else df.iloc[i]['ret_100']
            
            pred_dates.append(df.iloc[i]['trade_date'])
            strat_rets.append(strat_ret)
        
        if len(strat_rets) == 0:
            return params, -999, -999, -999, -999, -999, -999
        
        # 计算性能指标
        strat_rets = np.array(strat_rets)
        cum_ret = (1 + strat_rets).cumprod()
        total_ret = cum_ret[-1] - 1
        days = len(strat_rets)
        annual_ret = cum_ret[-1] ** (252 / days) - 1
        annual_vol = np.std(strat_rets) * np.sqrt(252)
        sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else -999
        
        # 计算最大回撤
        roll_max = np.maximum.accumulate(cum_ret)
        drawdowns = cum_ret / roll_max - 1
        max_drawdown = drawdowns.min()
        
        return params, annual_ret, total_ret, sharpe, annual_vol, days, max_drawdown
        
    except Exception as e:
        print(f"参数组合 {params} 出错: {e}")
        return params, -999, -999, -999, -999, -999, -999

def parameter_optimization(holding_days_range, window_train_range, max_workers=None, 
                         objective='annual_ret', risk_free_rate=0.02):
    """
    参数优化函数，使用多进程进行网格搜索（利用数据缓存）
    
    Args:
        holding_days_range: holding_days的取值区间，如range(10, 31, 5)
        window_train_range: window_train的取值区间，如range(40, 201, 20)
        max_workers: 进程数，默认为CPU核心数
        objective: 优化目标，'annual_ret', 'sharpe', 'calmar', 'multi_objective'
        risk_free_rate: 无风险利率，用于计算夏普比率
    """
    # 生成参数组合
    param_combinations = list(itertools.product(holding_days_range, window_train_range))
    
    # 设置进程数
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # 限制最大进程数为8，避免系统过载
    
    print(f"开始参数优化...")
    print(f"参数组合总数: {len(param_combinations)}")
    print(f"使用多进程处理，进程数: {max_workers}")
    print(f"优化目标: {objective}")
    print(f"holding_days范围: {min(holding_days_range)}-{max(holding_days_range)}")
    print(f"window_train范围: {min(window_train_range)}-{max(window_train_range)}")
    
    results = []
    failed_count = 0
    
    # 多进程处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        print(f"正在提交 {len(param_combinations)} 个任务到进程池...")
        future_to_params = {executor.submit(evaluate_parameter_combination, params): params 
                           for params in param_combinations}
        
        # 收集结果
        print(f"开始收集结果...")
        for i, future in enumerate(as_completed(future_to_params)):
            try:
                params, annual_ret, total_ret, sharpe, annual_vol, days, max_drawdown = future.result()
                
                if annual_ret > -999:  # 有效结果
                    results.append({
                        'holding_days': params[0],
                        'window_train': params[1],
                        'annual_ret': annual_ret,
                        'total_ret': total_ret,
                        'sharpe': sharpe,
                        'annual_vol': annual_vol,
                        'days': days,
                        'max_drawdown': max_drawdown
                    })
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                params = future_to_params[future]
                print(f"处理参数组合 {params} 时出错: {e}")
            
            # 显示进度
            if (i + 1) % 10 == 0:
                progress = (i + 1) / len(param_combinations) * 100
                print(f"已完成: {i+1}/{len(param_combinations)} ({progress:.1f}%), 成功: {len(results)}, 失败: {failed_count}")
        
        print(f"所有任务已完成，正在整理结果...")
    
    print(f"\n参数优化完成！")
    print(f"总组合数: {len(param_combinations)}")
    print(f"成功数: {len(results)}")
    print(f"失败数: {failed_count}")
    
    if len(results) == 0:
        print("没有找到有效的参数组合")
        return None
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算额外的指标
    results_df['calmar_ratio'] = results_df['annual_ret'] / abs(results_df['max_drawdown'])
    results_df['risk_adjusted_ret'] = results_df['annual_ret'] * (1 - abs(results_df['max_drawdown']))
    
    # 根据优化目标排序
    if objective == 'annual_ret':
        results_df = results_df.sort_values('annual_ret', ascending=False)
        print(f"\n按年化收益率排序")
    elif objective == 'sharpe':
        results_df = results_df.sort_values('sharpe', ascending=False)
        print(f"\n按夏普比率排序")
    elif objective == 'calmar':
        results_df = results_df.sort_values('calmar_ratio', ascending=False)
        print(f"\n按Calmar比率排序")
    elif objective == 'multi_objective':
        # 多目标优化：综合考虑收益率和回撤
        # 标准化指标到0-1范围
        results_df['annual_ret_norm'] = (results_df['annual_ret'] - results_df['annual_ret'].min()) / (results_df['annual_ret'].max() - results_df['annual_ret'].min())
        results_df['max_dd_norm'] = (results_df['max_drawdown'] - results_df['max_drawdown'].min()) / (results_df['max_drawdown'].max() - results_df['max_drawdown'].min())
        
        # 综合得分：收益率权重0.7，回撤权重0.3
        results_df['composite_score'] = 0.7 * results_df['annual_ret_norm'] - 0.3 * results_df['max_dd_norm']
        results_df = results_df.sort_values('composite_score', ascending=False)
        print(f"\n按综合得分排序（收益率70%，回撤30%）")
    
    print(f"有效结果数: {len(results_df)}")
    
    # 显示前10个最佳结果
    if objective == 'multi_objective':
        display_cols = ['holding_days', 'window_train', 'annual_ret', 'max_drawdown', 'composite_score', 'sharpe']
    else:
        display_cols = ['holding_days', 'window_train', 'annual_ret', 'max_drawdown', 'sharpe', 'annual_vol']
    
    print(f"\n前10个最佳参数组合:")
    print(results_df.head(10)[display_cols].to_string(index=False))
    
    # 最佳参数组合
    best_params = results_df.iloc[0]
    print(f"\n最佳参数组合:")
    print(f"holding_days: {best_params['holding_days']}")
    print(f"window_train: {best_params['window_train']}")
    print(f"年化收益率: {best_params['annual_ret']:.2%}")
    print(f"最大回撤: {best_params['max_drawdown']:.2%}")
    print(f"夏普比率: {best_params['sharpe']:.2f}")
    print(f"年化波动率: {best_params['annual_vol']:.2%}")
    if objective == 'multi_objective':
        print(f"综合得分: {best_params['composite_score']:.4f}")
    
    return results_df, best_params

# ====== 修改主函数调用 ======
if __name__ == '__main__':
    # 参数优化模式
    optimize_mode = True  # 设置为True进行参数优化，False进行单次回测
    
    if optimize_mode:
        # 参数优化
        print("=== 参数优化模式 ===")
        
        # 定义参数搜索范围
        holding_days_range = range(5, 51, 5)  # 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
        window_train_range = range(20, 401, 10)  # 20, 30, 40, ..., 400
        
        # 选择优化目标
        # 'annual_ret': 仅优化年化收益率
        # 'sharpe': 仅优化夏普比率
        # 'calmar': 仅优化Calmar比率（收益率/最大回撤）
        # 'multi_objective': 多目标优化（综合考虑收益率和回撤）
        objective = 'multi_objective'
        
        # 执行参数优化
        results_df, best_params = parameter_optimization(
            holding_days_range=holding_days_range,
            window_train_range=window_train_range,
            max_workers=6,  # 设置6个进程进行并行计算
            objective=objective
        )
        
        if results_df is not None:
            # 保存优化结果
            results_path = os.path.join(OUTPUT_DIR, f'parameter_optimization_results_{objective}.csv')
            results_df.to_csv(results_path, index=False)
            print(f"\n优化结果已保存到: {results_path}")
            
            # 使用最佳参数进行回测
            print(f"\n=== 使用最佳参数进行回测 ===")
            holding_days = int(best_params['holding_days'])
            window_train = int(best_params['window_train'])
        else:
            print("参数优化失败，使用默认参数")
            holding_days = 20
            window_train = 80
    else:
        # 单次回测模式
        print("=== 单次回测模式 ===")
        holding_days = 25
        window_train = 80
    
    # 准备数据
    df = prepare_logit_data(holding_days=holding_days)
    features = ['rsi_diff', 'vol_ratio_1m', 'mom_1m_diff', 'mom_3m_diff', 'cpi_yoy_mom_lag1','m2_yoy_mom_lag1']
    
    # 数据质量检查
    print("数据质量检查:")
    print(f"数据形状: {df.shape}")
    print(f"特征列: {features}")
    
    for feature in features:
        if feature in df.columns:
            inf_count = np.isinf(df[feature]).sum()
            nan_count = df[feature].isna().sum()
            print(f"{feature}: inf={inf_count}, nan={nan_count}")
    
    print("\n开始滚动窗口预测...")
    print(f"预测逻辑：用历史{window_train}天数据训练，对下一天进行预测")
    print(f"总数据天数：{len(df)}")
    print(f"训练窗口：{window_train}天")
    print(f"预测天数：{len(df) - window_train}天")
    
    df_oos = rolling_walk_forward(df, features, window_train=window_train)
    print("样本外窗口合并后数据：")
    print(df_oos.head())
    print(f"实际预测天数：{len(df_oos)}")
    
    # 样本外整体回测
    backtest_result = backtest(df_oos)
    
    # 保存训练配置和结果摘要
    config_summary = {
        'holding_days': holding_days,
        'window_train': window_train,
        'features': features,
        'optimize_mode': optimize_mode,
        'objective': objective if optimize_mode else 'single_test',
        'total_samples': len(df),
        'prediction_samples': len(df_oos),
        'start_date': df['trade_date'].min(),
        'end_date': df['trade_date'].max()
    }
    
    # 保存配置摘要
    import json
    config_path = os.path.join(OUTPUT_DIR, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_summary, f, ensure_ascii=False, indent=2)
    print(f"训练配置已保存到: {config_path}")