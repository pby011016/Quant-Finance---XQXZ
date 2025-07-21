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
    start_date = '20170101'
    end_date = '20250716'
    N = range(1, 60)  # 动量天数范围
    
    # 获取数据（只获取一次，在循环外）
    print("\n===== 获取大盘指数数据 =====")
    df_big = get_index_data('000903', start_date, end_date)  # 上证50
    
    print("\n===== 获取小盘指数数据 =====")
    df_small = get_index_data('000905', start_date, end_date)  # 中证1000

    # 检查数据是否有效
    if df_big.empty or df_small.empty:
        print("数据获取失败，无法继续执行策略")
        return
    
    # 计算大小盘每天的涨跌幅（在循环外计算一次）
    df_big['big_amp'] = df_big['close'] / df_big['close'].shift(1) - 1
    df_small['small_amp'] = df_small['close'] / df_small['close'].shift(1) - 1

    # 重命名列
    df_big.rename(columns={'open': 'big_open', 'close': 'big_close'}, inplace=True)
    df_small.rename(columns={'open': 'small_open', 'close': 'small_close'}, inplace=True)

    # 合并数据
    df_base = pd.merge(df_big, df_small, on='candle_end_time', how='inner')
    
    # 创建一个目录来保存所有结果
    results_dir = "大小盘轮动策略结果"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建一个列表来存储所有动量天数的结果
    results_list = []
    
    # 循环遍历所有动量天数
    for momentum_days in N:
        print(f"\n===== 开始计算策略（{momentum_days}日动量） =====")
        
        # 复制基础数据
        df = df_base.copy()
        
        # 计算N日的动量
        df['big_mom'] = df['big_close'].pct_change(periods=momentum_days)
        df['small_mom'] = df['small_close'].pct_change(periods=momentum_days)

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
        
        # 计算相对小盘指数的超额收益
        df['excess_return'] = df['strategy_net'] - df['small_net']
        
        # 计算策略表现指标
        last_net = df['strategy_net'].iloc[-1]
        last_small_net = df['small_net'].iloc[-1]
        excess_return = (last_net - last_small_net) * 100
        
        # 计算年化收益率
        days = (df['candle_end_time'].iloc[-1] - df['candle_end_time'].iloc[0]).days
        years = days / 365.25
        annual_return = (last_net ** (1/years) - 1) * 100 if years > 0 else 0
        small_annual_return = (last_small_net ** (1/years) - 1) * 100 if years > 0 else 0
        
        # 计算最大回撤
        df['peak'] = df['strategy_net'].cummax()
        df['drawdown'] = (df['strategy_net'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min() * 100
        
        # 计算交易次数
        trade_count = df['trade_time'].notnull().sum()
        
        # 计算夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        daily_returns = df['strategy_amp_adjust']
        annualized_return = (1 + daily_returns.mean()) ** 252 - 1
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
        
        # 计算盈亏比
        winning_trades = daily_returns[daily_returns > 0]
        losing_trades = daily_returns[daily_returns < 0]
        if len(losing_trades) > 0:
            profit_factor = winning_trades.sum() / abs(losing_trades.sum())
        else:
            profit_factor = np.inf
        
        # 保存结果到列表
        results_list.append({
            'momentum_days': momentum_days,
            'final_net_value': last_net,
            'small_net_value': last_small_net,
            'excess_return': excess_return,
            'annual_return': annual_return,
            'small_annual_return': small_annual_return,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor
        })
        
        # 创建当前动量天数的结果目录
        momentum_dir = os.path.join(results_dir, f"{momentum_days}日动量")
        os.makedirs(momentum_dir, exist_ok=True)
        
        # 保存当前动量天数的详细结果
        detailed_filename = os.path.join(momentum_dir, f"大小盘风格切换_{momentum_days}日动量.csv")
        df.to_csv(detailed_filename, encoding='gbk', index=False)
        
        # 保存净值数据
        net_value_filename = os.path.join(momentum_dir, f"大小盘风格切换_净值_{momentum_days}日动量.csv")
        df[['candle_end_time', 'strategy_amp_adjust', 'strategy_net', 'small_net', 'excess_return']].to_csv(net_value_filename, encoding='gbk', index=False)
        
        # 为当前动量天数绘制图表
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
        
        plt.title(f'大小盘风格轮动策略 ({momentum_days}日动量)', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加净值标签
        plt.annotate(f'策略净值: {last_net:.2f}\n小盘净值: {last_small_net:.2f}\n超额收益: {excess_return:.2f}%', 
                     xy=(df['candle_end_time'].iloc[-1], last_net),
                     xytext=(df['candle_end_time'].iloc[-1] - pd.Timedelta(days=180), last_net + 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=12)
        
        plt.tight_layout()

        # 保存图表
        plot_filename = os.path.join(momentum_dir, f"大小盘风格轮动策略_{momentum_days}日动量.jpg")
        plt.savefig(plot_filename, dpi=300)
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 打印当前动量天数的策略表现
        print(f"\n===== {momentum_days}日动量策略表现摘要 =====")
        print(f"策略起始日期: {df['candle_end_time'].iloc[0].strftime('%Y-%m-%d')}")
        print(f"策略结束日期: {df['candle_end_time'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"初始净值: 1.00")
        print(f"策略最终净值: {last_net:.2f}")
        print(f"小盘指数最终净值: {last_small_net:.2f}")
        print(f"相对小盘超额收益: {excess_return:.2f}%")
        print(f"策略总收益率: {(last_net - 1) * 100:.2f}%")
        print(f"小盘指数总收益率: {(last_small_net - 1) * 100:.2f}%")
        print(f"策略年化收益率: {annual_return:.2f}%")
        print(f"小盘指数年化收益率: {small_annual_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"交易次数: {trade_count}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"盈亏比: {profit_factor:.2f}")
    
    # 从列表创建DataFrame
    results_df = pd.DataFrame(results_list)
    
    # 保存所有动量天数的结果汇总
    summary_filename = os.path.join(results_dir, "所有动量天数策略表现汇总.csv")
    results_df.to_csv(summary_filename, encoding='gbk', index=False)
    
    # 找到表现最好的动量天数
    best_momentum = results_df.loc[results_df['final_net_value'].idxmax(), 'momentum_days']
    best_net_value = results_df.loc[results_df['final_net_value'].idxmax(), 'final_net_value']
    best_annual_return = results_df.loc[results_df['final_net_value'].idxmax(), 'annual_return']
    best_excess_return = results_df.loc[results_df['final_net_value'].idxmax(), 'excess_return']
    best_sharpe = results_df.loc[results_df['final_net_value'].idxmax(), 'sharpe_ratio']
    best_profit_factor = results_df.loc[results_df['final_net_value'].idxmax(), 'profit_factor']
    
    print("\n===== 最佳策略表现摘要 =====")
    print(f"最佳动量天数: {best_momentum}日")
    print(f"最终净值: {best_net_value:.4f}")
    print(f"年化收益率: {best_annual_return:.2f}%")
    print(f"相对小盘超额收益: {best_excess_return:.2f}%")
    print(f"夏普比率: {best_sharpe:.2f}")
    print(f"盈亏比: {best_profit_factor:.2f}")
    
    # 绘制所有动量天数的净值表现对比图
    plt.figure(figsize=(14, 8))
    
    # 为每个动量天数添加净值曲线
    for momentum_days in N:
        net_value_filename = os.path.join(results_dir, f"{momentum_days}日动量", f"大小盘风格切换_净值_{momentum_days}日动量.csv")
        if os.path.exists(net_value_filename):
            net_df = pd.read_csv(net_value_filename, encoding='gbk', parse_dates=['candle_end_time'])
            plt.plot(net_df['candle_end_time'], net_df['strategy_net'], label=f'{momentum_days}日动量')
    
    plt.title('不同动量天数策略表现对比', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('净值', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存对比图
    comparison_filename = os.path.join(results_dir, "不同动量天数策略表现对比.jpg")
    plt.savefig(comparison_filename, dpi=300)
    plt.close()
    
    # 绘制超额收益对比图
    plt.figure(figsize=(14, 8))
    
    for momentum_days in N:
        net_value_filename = os.path.join(results_dir, f"{momentum_days}日动量", f"大小盘风格切换_净值_{momentum_days}日动量.csv")
        if os.path.exists(net_value_filename):
            net_df = pd.read_csv(net_value_filename, encoding='gbk', parse_dates=['candle_end_time'])
            plt.plot(net_df['candle_end_time'], net_df['excess_return'], label=f'{momentum_days}日动量')
    
    plt.title('不同动量天数超额收益对比', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('超额收益(相对于小盘指数)', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存超额收益对比图
    excess_comparison_filename = os.path.join(results_dir, "不同动量天数超额收益对比.jpg")
    plt.savefig(excess_comparison_filename, dpi=300)
    plt.close()
    
    print(f"\n所有策略结果已保存到目录: {results_dir}")
    print(f"策略表现汇总已保存: {summary_filename}")
    print(f"策略对比图已保存: {comparison_filename}")
    print(f"超额收益对比图已保存: {excess_comparison_filename}")

    # 绘制图片
    plt.plot(df['candle_end_time'], df['strategy_net'], label='strategy')
    plt.plot(df['candle_end_time'], df['big_net'], label='big_net')
    plt.plot(df['candle_end_time'], df['small_net'], label='small_net')
    plt.legend()
    plt.show()

# ================== 主执行函数 ==================
if __name__ == "__main__":
    print("===== 开始执行大小盘风格轮动策略 =====")
    run_strategy()
    print("===== 策略执行完成 =====")