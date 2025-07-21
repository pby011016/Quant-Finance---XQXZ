import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import os
import datetime
import pickle
import hashlib

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置缓存目录
CACHE_DIR = "tushare_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 初始化Tushare
try:
    pro = ts.pro_api('4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77')  # 替换为您的Tushare token
    print("Tushare初始化成功")
except Exception as e:
    print(f"Tushare初始化失败: {e}")
    raise

# ================== 缓存工具函数 ==================
def get_cache_filename(index_code, start_date, end_date):
    """生成缓存文件名"""
    params_str = f"{index_code}_{start_date}_{end_date}"
    return f"{hashlib.md5(params_str.encode()).hexdigest()}.pkl"

def save_to_cache(filename, data):
    """保存数据到缓存"""
    filepath = os.path.join(CACHE_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"数据已缓存: {filepath}")

def load_from_cache(filename):
    """从缓存加载数据"""
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"从缓存加载数据: {filepath}")
        return data
    return None

# ================== 数据获取函数 ==================
def get_index_data(index_code, start_date, end_date):
    """从Tushare获取指数数据，带缓存功能"""
    # 生成缓存文件名
    cache_filename = get_cache_filename(index_code, start_date, end_date)
    
    # 尝试从缓存加载
    cached_data = load_from_cache(cache_filename)
    if cached_data is not None:
        return cached_data
    
    print(f"从Tushare获取数据: {index_code} ({start_date}至{end_date})")
    
    # 定义市场后缀
    market = '.SH' if index_code.startswith('00') else '.SZ'
    
    # 获取数据
    try:
        df = pro.index_daily(ts_code=index_code + market, 
                             start_date=start_date, 
                             end_date=end_date)
        
        # 数据处理
        if not df.empty:
            df = df[['trade_date', 'open', 'close']]
            df.rename(columns={'trade_date': 'candle_end_time', 'open': 'open', 'close': 'close'}, inplace=True)
            df['candle_end_time'] = pd.to_datetime(df['candle_end_time'])
            df.sort_values('candle_end_time', inplace=True)
            
            # 保存到缓存
            save_to_cache(cache_filename, df)
            return df
        else:
            print(f"未获取到数据: {index_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"获取数据失败: {index_code} - {e}")
        return pd.DataFrame()

# ================== 策略执行函数 ==================
def run_strategy():
    # 设置参数
    trade_rate = 1 / 10000  # 场内基金万分之0.6，买卖手续费相同，无印花税
    # momentum_days = 20  # 计算20天的动量
    # start_date = '20190101'  # 扩大到2019年开始
    start_date = '20200101'  # 从2022年开始
    end_date = '20250718'  # 到2025年结束

    # 获取数据
    print("\n===== 获取大盘指数数据 =====")
    # 大盘：上证50(000016) / 沪深300(000300) / 中证500(000905)
    df_big = get_index_data('000016', start_date, end_date)  # 上证50

    print("\n===== 获取小盘指数数据 =====")
    # 小盘：中证500(000905) / 中证1000(000852) / 创业板(399006) / 科创50(000688) / 创业200(399019)
    df_small = get_index_data('000852', start_date, end_date)  # 中证500

    # 检查数据是否有效
    if df_big.empty or df_small.empty:
        print("数据获取失败，无法继续执行策略")
        return
    
    print("\n===== 开始计算策略 =====")
    
    # 计算大小盘每天的涨跌幅
    df_big['big_amp'] = df_big['close'] / df_big['close'].shift(1) - 1
    df_small['small_amp'] = df_small['close'] / df_small['close'].shift(1) - 1

    # 重命名列
    df_big.rename(columns={'open': 'big_open', 'close': 'big_close'}, inplace=True)
    df_small.rename(columns={'open': 'small_open', 'close': 'small_close'}, inplace=True)

    # 合并数据
    df = pd.merge(df_big, df_small, on='candle_end_time', how='inner')

    # 计算20日的动量
    df['big_mom'] = df['big_close'].pct_change(periods=20)
    df['small_mom'] = df['small_close'].pct_change(periods=20)

    # 风格变换条件
    df.loc[df['big_mom'] > df['small_mom'], 'style'] = 'big'
    df.loc[df['big_mom'] < df['small_mom'], 'style'] = 'small'
    df.loc[(df['big_mom'] < 0) & (df['small_mom'] < 0), 'style'] = 'empty'

    # 填充缺失值（维持前一日持仓）
    df['style'] = df['style'].ffill()

    # 调整持仓日期
    df['pos'] = df['style'].shift(1)

    # 删除持仓为nan的天数
    df.dropna(subset=['pos'], inplace=True)

    # 计算策略的整体涨跌幅
    df.loc[df['pos'] == 'big', 'strategy_amp'] = df['big_amp']
    df.loc[df['pos'] == 'small', 'strategy_amp'] = df['small_amp']
    df.loc[df['pos'] == 'empty', 'strategy_amp'] = 0

    # 调仓时间
    df.loc[df['pos'] != df['pos'].shift(1), 'trade_time'] = df['candle_end_time']

    # 处理调仓日成本
    df['strategy_amp_adjust'] = df['strategy_amp']  # 非调仓日直接使用原始涨跌幅

    # 买入日的涨跌幅（考虑开盘价和交易成本）
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == 'big'), 'strategy_amp_adjust'] = df['big_close'] / (df['big_open'] * (1 + trade_rate)) - 1
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == 'small'), 'strategy_amp_adjust'] = df['small_close'] / (df['small_open'] * (1 + trade_rate)) - 1

    # 卖出时的交易成本
    df.loc[df['trade_time'].shift(-1).notnull() & (df['pos'] != 'empty'), 'strategy_amp_adjust'] = (1 + df['strategy_amp_adjust']) * (1 - trade_rate) - 1

    # 计算净值
    df['big_net'] = df['big_close'] / df['big_close'].iloc[0]
    df['small_net'] = df['small_close'] / df['small_close'].iloc[0]
    df['strategy_net'] = (1 + df['strategy_amp_adjust']).cumprod()

    # 绘制图表
    plt.figure(figsize=(14, 8))
    plt.plot(df['candle_end_time'], df['strategy_net'], label='轮动策略', linewidth=2.5, color='blue')
    plt.plot(df['candle_end_time'], df['big_net'], label='上证50', alpha=0.8, color='green')
    plt.plot(df['candle_end_time'], df['small_net'], label='创业200', alpha=0.8, color='red')
    
    # 添加策略信号点
    trade_dates = df[df['trade_time'].notnull()]['candle_end_time']
    trade_net = df[df['trade_time'].notnull()]['strategy_net']
    plt.scatter(trade_dates, trade_net, color='purple', s=50, label='调仓点', zorder=5)
    
    # 添加空仓区域
    empty_periods = df[df['pos'] == 'empty']
    if not empty_periods.empty:
        for i in range(0, len(empty_periods)-1):
            if empty_periods.index[i+1] - empty_periods.index[i] == 1:
                start_date = empty_periods.iloc[i]['candle_end_time']
                end_date = empty_periods.iloc[i+1]['candle_end_time']
                plt.axvspan(start_date, end_date, color='gray', alpha=0.2)
    
    # plt.title(f'大小盘风格轮动策略 ({start_date}至{end_date})', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('净值', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加净值标签
    last_net = df['strategy_net'].iloc[-1]
    plt.annotate(f'最终净值: {last_net:.2f}', 
                 xy=(df['candle_end_time'].iloc[-1], last_net),
                 xytext=(df['candle_end_time'].iloc[-1] - pd.Timedelta(days=180), last_net + 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)
    
    plt.tight_layout()

    # 保存图表
    plt.savefig('大小盘风格轮动策略.jpg', dpi=300)
    print("\n图表已保存为：大小盘风格轮动策略.jpg")

    # 保存结果
    df.to_csv('大小盘风格切换_改进.csv', encoding='gbk', index=False)
    df[['candle_end_time', 'strategy_amp_adjust', 'strategy_net']].to_csv('大小盘风格切换_净值.csv', encoding='gbk', index=False)
    print("结果数据已保存")

    # 显示图表
    plt.show()
    
    # 打印策略表现
    print("\n===== 策略表现摘要 =====")
    print(f"策略起始日期: {df['candle_end_time'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"策略结束日期: {df['candle_end_time'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"初始净值: 1.00")
    print(f"最终净值: {last_net:.2f}")
    print(f"总收益率: {(last_net - 1) * 100:.2f}%")
    
    # 计算年化收益率
    days = (df['candle_end_time'].iloc[-1] - df['candle_end_time'].iloc[0]).days
    years = days / 365.25
    annual_return = (last_net ** (1/years) - 1) * 100 if years > 0 else 0
    print(f"年化收益率: {annual_return:.2f}%")
    
    # 计算最大回撤
    df['peak'] = df['strategy_net'].cummax()
    df['drawdown'] = (df['strategy_net'] - df['peak']) / df['peak']
    max_drawdown = df['drawdown'].min() * 100
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    # 计算交易次数
    trade_count = df['trade_time'].notnull().sum()
    print(f"交易次数: {trade_count}")

# ================== 主执行函数 ==================
if __name__ == "__main__":
    print("===== 开始执行大小盘风格轮动策略 =====")
    run_strategy()
    print("===== 策略执行完成 =====")