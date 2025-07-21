import pandas as pd
import numpy as np
import tushare as ts
import sqlite3
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据获取与缓存模块 (修复版)
class DataFetcher:
    def __init__(self, token):
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.db_path = "quant_data.db"
        
        # 删除旧数据库文件
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print("已删除旧数据库文件，将创建新数据库")
        
        self._init_db()
        
    def _init_db(self):
        """初始化数据库 - 确保表结构正确"""
        with sqlite3.connect(self.db_path) as conn:
            # 删除旧表（如果存在）
            conn.execute("DROP TABLE IF EXISTS index_daily")
            
            # 创建新表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS index_daily (
                    ts_code TEXT, 
                    trade_date TEXT, 
                    open REAL, 
                    high REAL,
                    low REAL,
                    close REAL, 
                    vol REAL, 
                    amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )''')
    
    def fetch_data(self, ts_code, start_date, end_date):
        """获取并缓存数据"""
        with sqlite3.connect(self.db_path) as conn:
            # 检查缓存
            query = f"""
                SELECT * FROM index_daily 
                WHERE ts_code='{ts_code}' 
                AND trade_date BETWEEN '{start_date}' AND '{end_date}'
            """
            cached = pd.read_sql(query, conn)
            if not cached.empty:
                return cached
            
            # 从Tushare获取 - 添加重试机制
            for retry in range(3):  # 最多重试3次
                try:
                    print(f"正在获取 {ts_code} {start_date} 至 {end_date} 数据...")
                    df = self.pro.index_daily(
                        ts_code=ts_code, 
                        start_date=start_date, 
                        end_date=end_date
                    )
                    
                    # 确保有数据
                    if df.empty:
                        print(f"警告：{ts_code} 无数据返回")
                        return df
                    
                    # 简化字段，只保留必要数据
                    simple_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
                    df = df[simple_columns]
                    
                    # 缓存数据
                    df.to_sql('index_daily', conn, if_exists='append', index=False)
                    print(f"成功缓存 {len(df)} 条数据")
                    return df
                except Exception as e:
                    wait_time = 10 * (retry + 1)
                    print(f"获取数据失败: {e}, {wait_time}秒后重试...")
                    time.sleep(wait_time)
            print(f"错误：无法获取 {ts_code} 数据")
            return pd.DataFrame()  # 返回空DataFrame防止程序崩溃


# 2. 因子计算模块
class FactorCalculator:
    @staticmethod
    def calculate_rsi(df, window=14):
        """计算RSI指标"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        # 避免除以零错误
        avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_psy(df, window):
        """计算PSY指标"""
        returns = df['close'].pct_change()
        # 使用循环避免在窗口期不足时的错误
        psy_values = []
        for i in range(len(returns)):
            start_idx = max(0, i - window + 1)
            if pd.isna(returns.iloc[start_idx:i+1]).any():
                psy_values.append(np.nan)
            else:
                up_count = (returns.iloc[start_idx:i+1] > 0).sum()
                psy_values.append(up_count / window * 100)
        return pd.Series(psy_values, index=df.index)

# 3. 策略回测模块
class StyleRotationStrategy:
    def __init__(self, token):
        self.fetcher = DataFetcher(token)
        self.large_cap = "000903.SH"  # 中证100
        self.small_cap = "000905.SH"  # 中证500
        
    def prepare_data(self, start_date, end_date):
        """准备大盘和小盘数据"""
        large = self.fetcher.fetch_data(self.large_cap, start_date, end_date)
        small = self.fetcher.fetch_data(self.small_cap, start_date, end_date)
        
        # 检查数据是否获取成功
        if large.empty or small.empty:
            raise ValueError("数据获取失败，请检查Tushare token或网络连接")

        # 合并数据
        merged = pd.merge(large, small, on='trade_date', 
                          suffixes=('_large', '_small'))
        # 合并后显式按日期升序排序
        merged = merged.sort_values('trade_date', ascending=True)  # 确保升序排列
        
        # 设置日期索引
        merged['trade_date'] = pd.to_datetime(merged['trade_date'])
        merged.set_index('trade_date', inplace=True)
        
        return merged
        # # 合并数据
        # merged = pd.merge(large, small, on='trade_date', 
        #                   suffixes=('_large', '_small'))
        # merged['trade_date'] = pd.to_datetime(merged['trade_date'])
        # merged.set_index('trade_date', inplace=True)
        # return merged
    
    def calculate_features(self, data):
        """计算四因子特征"""
        # 1. 3月RSI差值 (动量因子)
        data['rsi_3m_large'] = FactorCalculator.calculate_rsi(data[['close_large']].rename(columns={'close_large': 'close'}), 60)
        data['rsi_3m_small'] = FactorCalculator.calculate_rsi(data[['close_small']].rename(columns={'close_small': 'close'}), 60)
        data['rsi_diff'] = data['rsi_3m_small'] - data['rsi_3m_large']
        
        # 2. 6月PSY差值 (反转因子)
        data['psy_6m_large'] = FactorCalculator.calculate_psy(data[['close_large']].rename(columns={'close_large': 'close'}), 6)
        data['psy_6m_small'] = FactorCalculator.calculate_psy(data[['close_small']].rename(columns={'close_small': 'close'}), 6)
        data['psy_6m_diff'] = data['psy_6m_small'] - data['psy_6m_large']
        
        # 3. 12月PSY差值 (反转因子)
        data['psy_12m_large'] = FactorCalculator.calculate_psy(data[['close_large']].rename(columns={'close_large': 'close'}), 12)
        data['psy_12m_small'] = FactorCalculator.calculate_psy(data[['close_small']].rename(columns={'close_small': 'close'}), 12)
        data['psy_12m_diff'] = data['psy_12m_small'] - data['psy_12m_large']
        
        # 4. 1月成交量比值 (反转因子)
        data['vol_ratio'] = data['vol_small'] / data['vol_large']
        
        # 下月收益率差 (标签)
        data['small_ret'] = np.log(data['close_small']).diff()
        data['large_ret'] = np.log(data['close_large']).diff()
        data['next_ret'] = data['small_ret'].shift(-1) - data['large_ret'].shift(-1)
        data['label'] = (data['next_ret'] > 0).astype(int)
        
        # 清理数据
        return data.dropna()
    
    def backtest(self, start_date, end_date):
        """滚动回测策略"""
        full_data = self.prepare_data(start_date, end_date)
        features = self.calculate_features(full_data.copy())
        
        # 特征选择
        X = features[['rsi_diff', 'psy_6m_diff', 'psy_12m_diff', 'vol_ratio']]
        y = features['label']
        
        # 滚动训练和预测
        predictions = []
        probabilities = []
        
        # 滚动窗口：3年数据训练，预测下个月
        for i in range(36, len(X)):
            train_X = X.iloc[i-36:i-1]
            train_y = y.iloc[i-36:i-1]
            test_X = X.iloc[[i]]
            
            # 确保数据没有NaN
            if train_X.isna().any().any() or train_y.isna().any() or test_X.isna().any().any():
                # 数据不足，跳过本月
                predictions.append(np.nan)
                probabilities.append(np.nan)
                continue
            
            # 标准化
            scaler = StandardScaler()
            scaled_train_X = scaler.fit_transform(train_X)
            scaled_test_X = scaler.transform(test_X)
            
            # 训练Logit模型
            model = LogisticRegression()
            model.fit(scaled_train_X, train_y)
            
            # 预测
            pred = model.predict(scaled_test_X)[0]
            prob = model.predict_proba(scaled_test_X)[0][1]
            
            predictions.append(pred)
            probabilities.append(prob)
        
        # 合并结果
        results = features.iloc[36:].copy()
        results['pred'] = predictions
        results['prob'] = probabilities
        
        # 清洗预测数据
        results = results.dropna(subset=['pred'])
        
        # 计算策略收益
        # 配对交易策略 (多空)
        results['strategy_ret'] = np.where(
            results['pred'] == 1, 
            results['small_ret'] - results['large_ret'],
            results['large_ret'] - results['small_ret']
        )
        
        # 配置策略(仅做多)
        results['config_ret'] = np.where(
            results['pred'] == 1,
            results['small_ret'],
            results['large_ret']
        )
        
        # 计算累计收益
        results['cum_strategy'] = (1 + results['strategy_ret']).cumprod()
        results['cum_config'] = (1 + results['config_ret']).cumprod()
        
        # 获取沪深300数据作为基准
        hs300 = self.fetcher.fetch_data("000300.SH", start_date, end_date)
        hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
        hs300.set_index('trade_date', inplace=True)
        hs300['ret'] = hs300['close'].pct_change()
        hs300['cum_hs300'] = (1 + hs300['ret']).cumprod()
        
        # 合并基准数据
        results = results.merge(hs300[['cum_hs300']], left_index=True, right_index=True, how='left')
        
        return results
    
    def evaluate(self, results):
        """策略评估"""
        # 计算指标
        if len(results) == 0:
            print("无有效回测结果")
            return
            
        accuracy = (results['pred'] == results['label']).mean()
        total_ret = {
            'cum_strategy': results['cum_strategy'].iloc[-1] - 1,
            'cum_config': results['cum_config'].iloc[-1] - 1,
            'cum_hs300': results['cum_hs300'].iloc[-1] - 1
        }
        
        # 计算最大回撤
        strategy_dd = (results['cum_strategy'].cummax() - results['cum_strategy'])/results['cum_strategy'].cummax()
        config_dd = (results['cum_config'].cummax() - results['cum_config'])/results['cum_config'].cummax()

        # 1. 年化收益率
        years = len(results) / 52  # 周频数据，52周/年
        annualized_strategy = (results['cum_strategy'].iloc[-1])**(1/years) - 1
        annualized_config = (results['cum_config'].iloc[-1])**(1/years) - 1
        
        # 2. 夏普比率（假设无风险利率为0）
        strategy_sharpe = results['strategy_ret'].mean() / results['strategy_ret'].std() * np.sqrt(52)
        config_sharpe = results['config_ret'].mean() / results['config_ret'].std() * np.sqrt(52)
        
        # 3. 盈亏比
        strategy_wins = results[results['strategy_ret'] > 0]['strategy_ret'].sum()
        strategy_losses = abs(results[results['strategy_ret'] < 0]['strategy_ret'].sum())
        strategy_profit_ratio = strategy_wins / strategy_losses if strategy_losses > 0 else np.inf
        
        config_wins = results[results['config_ret'] > 0]['config_ret'].sum()
        config_losses = abs(results[results['config_ret'] < 0]['config_ret'].sum())
        config_profit_ratio = config_wins / config_losses if config_losses > 0 else np.inf        
        
        # 打印结果
        print("\n===== 策略评估结果 =====")
        print(f"回测周期: {results.index[0].date()} 至 {results.index[-1].date()}")
        print(f"总样本数: {len(results)}")
        print(f"策略准确率: {accuracy:.2%}")
        print(f"\n累计收益率:")
        print(f"  配对策略: {total_ret['cum_strategy']:.2%}")
        print(f"  配置策略: {total_ret['cum_config']:.2%}")
        print(f"  沪深300: {total_ret['cum_hs300']:.2%}")
        print(f"\n最大回撤:")
        print(f"  配对策略: {strategy_dd.max():.2%}")
        print(f"  配置策略: {config_dd.max():.2%}")
        print(f"年化收益率:")
        print(f"  配对策略: {annualized_strategy:.2%}")
        print(f"  配置策略: {annualized_config:.2%}")
        print(f"\n夏普比率:")
        print(f"  配对策略: {strategy_sharpe:.2f}")
        print(f"  配置策略: {config_sharpe:.2f}")
        print(f"\n盈亏比:")
        print(f"  配对策略: {strategy_profit_ratio:.2f}")
        print(f"  配置策略: {config_profit_ratio:.2f}")
            
        # 可视化
        self.plot_results(results)
    
    def plot_results(self, results):
        """可视化结果"""
        if len(results) == 0:
            print("无数据可可视化")
            return
            
        plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2, 2, 2])
        
        # 累计收益
        ax1 = plt.subplot(gs[0])
        results[['cum_strategy', 'cum_config', 'cum_hs300']].plot(ax=ax1, title='累计收益对比')
        ax1.set_ylabel('累计收益')
        ax1.legend(['多空配对策略', '多头配置策略', '沪深300基准'])
        
        # 预测概率
        ax2 = plt.subplot(gs[1])
        results['prob'].plot(ax=ax2, color='purple', title='小盘跑赢概率预测')
        ax2.axhline(0.5, color='r', linestyle='--')
        ax2.set_ylabel('概率')
        
        # 因子变化
        ax3 = plt.subplot(gs[2])
        results[['rsi_diff', 'psy_6m_diff', 'psy_12m_diff']].plot(ax=ax3, title='因子走势')
        ax3.set_ylabel('因子值')
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.legend(['RSI差值(小盘-大盘)', '6月PSY差值', '12月PSY差值'])
        
        # 成交量比率
        ax4 = plt.subplot(gs[3])
        results['vol_ratio'].plot(ax=ax4, color='orange', title='成交量比率(小盘/大盘)')
        ax4.set_ylabel('比率')
        
        plt.tight_layout()
        plt.savefig('style_rotation_results.png', dpi=300)
        plt.show()

# 主函数 - 添加更多错误处理
if __name__ == "__main__":
    TOKEN = "4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77"
    
    print("开始运行大小盘轮动策略...")
    print("="*50)
    
    try:
        strategy = StyleRotationStrategy(TOKEN)
        
        # 样本内区间 (2007-02至2011-12)
        sample_start = "20070201"
        sample_end = "20111231"
        
        # 样本外推1年 (2012年)
        out_sample_start = "20120101"
        out_sample_end = "20121231"
        
        # 样本内训练
        print("===== 样本内训练 (2007-02至2011-12) =====")
        sample_results = strategy.backtest(sample_start, sample_end)
        strategy.evaluate(sample_results)
        
        # 样本外测试
        print("\n===== 样本外测试 (2012年) =====")
        out_sample_results = strategy.backtest(out_sample_start, out_sample_end)
        strategy.evaluate(out_sample_results)
        
    except Exception as e:
        # ... [错误处理代码] ...
# if __name__ == "__main__":
#     TOKEN = "4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77"  # 替换为你的Tushare token
    
#     print("开始运行大小盘轮动策略...")
#     print("="*50)
    
#     try:
#         strategy = StyleRotationStrategy(TOKEN)
#         results = strategy.backtest(start_date="20100101", end_date="20120831")
#         strategy.evaluate(results)
#     except Exception as e:
#         print(f"\n❌ 策略运行失败: {e}")
#         print("可能的原因：")
#         print("- Tushare token 无效或过期")
#         print("- 网络连接问题")
#         print("- 数据获取限制")
#         print("\n建议解决方案：")
#         print("1. 检查并更新你的Tushare token")
#         print("2. 确认网络连接正常")
#         print("3. 尝试更小的回测时间范围（如20150101-20201231）")
#         print("4. 等待后重试（Tushare有限流）")
        
        import traceback
        traceback.print_exc()
        print("\n如需进一步帮助，请提供完整错误信息")

