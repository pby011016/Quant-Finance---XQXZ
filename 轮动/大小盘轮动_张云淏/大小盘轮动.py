import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 设置Tushare
# ts.set_token("66de44d6dc04f52b904af7f91e8e279cc418b843ae110c704a2358a2")
pro = ts.pro_api("4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77")
# pro = ts.pro_api()

# 获取上证50指数数据
sh50_data = pro.index_daily(ts_code='000016.SH', start_date='20200101', end_date='20241025')
sh50_data = sh50_data[['trade_date', 'close']].rename(columns={'trade_date': '日期', 'close': '收盘价_sh50'})

# 获取创业板指数据
cyb_data = pro.index_daily(ts_code='399006.SZ', start_date='20200101', end_date='20241025')
cyb_data = cyb_data[['trade_date', 'close']].rename(columns={'trade_date': '日期', 'close': '收盘价_cyb'})

# 获取沪深300指数数据
hs300_data = pro.index_daily(ts_code='000300.SH', start_date='20200101', end_date='20241025')
hs300_data = hs300_data[['trade_date', 'close']].rename(columns={'trade_date': '日期', 'close': '收盘价_hs300'})


# 合并数据
data = pd.merge(sh50_data, cyb_data, on='日期')
data = pd.merge(data, hs300_data, on='日期')  # 添加沪深300数据
data['日期'] = pd.to_datetime(data['日期'])  # 转换日期格式
data.sort_values('日期', inplace=True)  # 按日期排序

# 初始化投资
initial_investment = 10000  # 总投资额
half_investment = initial_investment / 2  # 各持有一半

# 初始化持仓
data['cash'] = initial_investment  # 初始现金
data['total_value'] = initial_investment  # 总资产
data['position'] = 0  # 持仓状态，0: 不持仓, 1: 持仓大盘, -1: 持仓小盘

# 初始时，购买上证50和创业板指各一半
data['position'].iloc[0] = 1  # 假设持有上证50
data['cash'].iloc[0] -= data['收盘价_sh50'].iloc[0]  # 扣除上证50购买金额
data['cash'].iloc[0] -= data['收盘价_cyb'].iloc[0]  # 扣除创业板指购买金额
data['total_value'].iloc[0] = data['cash'].iloc[0] + (data['收盘价_sh50'].iloc[0] + data['收盘价_cyb'].iloc[0])

# 计算指标 R
data['R'] = np.log(data['收盘价_cyb']) - np.log(data['收盘价_sh50'])

# 计算过去10日均值与120日的均值和标准差
data['R_mean_10'] = data['R'].rolling(window=10).mean()
data['R_mean_120'] = data['R'].rolling(window=60).mean()
data['R_std'] = data['R'].rolling(window=60).std(ddof=0)

# 定义三条均线
data['high_line'] = data['R_mean_120'] + 0.8 * data['R_std']  # 高均线
data['mid_line'] = data['R_mean_10']  # 中线
data['low_line'] = data['R_mean_120'] - 0.8 * data['R_std']  # 低均线

# 交易信号初始化
data['final_signal'] = 0  # 最终交易信号

# 生成交易信号
for i in range(1, len(data)):
    # A: 回落 (中线从上方向下穿高均线)
    if data['mid_line'].iloc[i] <= data['high_line'].iloc[i] and data['mid_line'].iloc[i - 1] > data['high_line'].iloc[i - 1]:
        data['final_signal'].iloc[i] = 1  # 买入大盘，卖出小盘

    # B: 爬升 (中线从下方向上穿高均线)
    elif data['mid_line'].iloc[i] >= data['high_line'].iloc[i] and data['mid_line'].iloc[i - 1] < data['high_line'].iloc[i - 1]:
        data['final_signal'].iloc[i] = -1  # 卖出大盘，买入小盘

    # C: 跌坠 (中线从上方向下穿低均线)
    elif data['mid_line'].iloc[i] <= data['low_line'].iloc[i] and data['mid_line'].iloc[i - 1] > data['low_line'].iloc[i - 1]:
        data['final_signal'].iloc[i] = 1  # 买入大盘，卖出小盘

    # D: 复苏 (中线从下方向上穿低均线)
    elif data['mid_line'].iloc[i] >= data['low_line'].iloc[i] and data['mid_line'].iloc[i - 1] < data['low_line'].iloc[i - 1]:
        data['final_signal'].iloc[i] = -1  # 卖出大盘，买入小盘

# 回测计算
for i in range(1, len(data)):
    # 当前持仓
    previous_position = data['position'].iloc[i - 1]

    if data['final_signal'].iloc[i] == 1:  # 买入大盘，卖出小盘
        if previous_position != 1:  # 如果之前没有持仓大盘
            # 清仓小盘（如果有持仓小盘）
            if previous_position == -1:
                data['cash'].iloc[i] += data['收盘价_cyb'].iloc[i]  # 加上小盘出售金额

            # 买入大盘
            data['position'].iloc[i] = 1
            data['cash'].iloc[i] -= data['收盘价_sh50'].iloc[i]  # 扣除大盘购买金额

        else:
            data['position'].iloc[i] = 1  # 继续持有大盘

    elif data['final_signal'].iloc[i] == -1:  # 卖出大盘，买入小盘
        if previous_position != -1:  # 如果之前没有持仓小盘
            # 清仓大盘（如果有持仓大盘）
            if previous_position == 1:
                data['cash'].iloc[i] += data['收盘价_sh50'].iloc[i]  # 加上大盘出售金额

            # 买入小盘
            data['position'].iloc[i] = -1
            data['cash'].iloc[i] -= data['收盘价_cyb'].iloc[i]  # 扣除小盘购买金额

        else:
            data['position'].iloc[i] = -1  # 继续持有小盘

    else:  # 其他情况保持不变
        data['position'].iloc[i] = previous_position

    # 更新总资产
    if data['position'].iloc[i] == 1:  # 如果持有大盘
        data['total_value'].iloc[i] = data['cash'].iloc[i] + data['收盘价_sh50'].iloc[i]
    elif data['position'].iloc[i] == -1:  # 如果持有小盘
        data['total_value'].iloc[i] = data['cash'].iloc[i] + data['收盘价_cyb'].iloc[i]
    else:
        data['total_value'].iloc[i] = data['cash'].iloc[i]  # 现金持有

# 计算收益率
data['returns'] = data['total_value'].pct_change()

# 检查最后几天的总资产和收益率
print(data[['日期', 'total_value', 'returns']].tail(10))  # 输出最后10天的资产和收益率

# 计算沪深300收益率
data['hs300_returns'] = data['收盘价_hs300'].pct_change()  # 沪深300收益率

# 计算最大回撤
data['max_drawdown'] = (data['total_value'] / data['total_value'].cummax() - 1)
max_drawdown_value = data['max_drawdown'].min()

# 计算年化收益率
annualized_return = (1 + data['returns'].mean()) ** 252 - 1  # 年化收益率

# 计算信息比率
benchmark_returns = data['hs300_returns']  # 沪深300收益率
excess_returns = data['returns'] - benchmark_returns  # 超额收益率
information_ratio = excess_returns.mean() / excess_returns.std()

# 计算夏普率
sharpe_ratio = (data['returns'].mean() / data['returns'].std()) * np.sqrt(252)

# 计算索丁诺比率
sortino_ratio = (data['returns'].mean() / data['returns'][data['returns'] < 0].std()) * np.sqrt(252)

# 输出结果
print(f"最终总资产: {data['total_value'].iloc[-1]:.2f}元")
print(f"年化收益率: {annualized_return * 100:.2f}%")
print(f"沪深300年化收益率: {(data['hs300_returns'].mean() * 252) * 100:.2f}%")
print(f"最大回撤: {max_drawdown_value * 100:.2f}%")
print(f"信息比率: {information_ratio:.2f}")
print(f"夏普率: {sharpe_ratio:.2f}")
print(f"索丁诺比率: {sortino_ratio:.2f}")

# 可视化三条线及买入卖出点
plt.figure(figsize=(14, 7))
plt.plot(data['日期'], data['high_line'], label='高均线', color='blue')
plt.plot(data['日期'], data['mid_line'], label='中线', color='red')
plt.plot(data['日期'], data['low_line'], label='低均线', color='yellow')

# 绘制买入卖出信号点
buy_signals = data[data['final_signal'] == 1]  # 买入信号
sell_signals = data[data['final_signal'] == -1]  # 卖出信号

# 标记买入点
plt.scatter(buy_signals['日期'], buy_signals['mid_line'], marker='^', color='green', label='买入信号', s=100)
# 标记卖出点
plt.scatter(sell_signals['日期'], sell_signals['mid_line'], marker='v', color='purple', label='卖出信号', s=100)

# 添加图例和标题
plt.title('三条均线及买入卖出信号')
plt.xlabel('日期')
plt.ylabel('点位')
plt.axhline(0, color='black', lw=1)  # 添加水平线
plt.legend()
plt.grid()
plt.xticks(rotation=45)  # 旋转日期标签以便更好地显示
plt.tight_layout()  # 自动调整布局以避免重叠
plt.show()






