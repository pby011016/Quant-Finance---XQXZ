import pandas as pd
import numpy as np
from function import *
import matplotlib.pyplot as plt

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

# 读取数据
# 大盘： 上证50（000016）/沪深300（000300）/中证500（000905）
# 小盘： 中证500（000905）/中证1000（000852）/创业板（399006）/科创50（000688）
df = pd.read_csv('大小盘风格切换_净值.csv', encoding='gbk', parse_dates=['candle_end_time'])
df['收盘价_复权'] = df['strategy_net']
df['收盘价'] = df['收盘价_复权']
df['前收盘价'] = df['收盘价_复权'].shift(1)

# =====计算移动平均线策略的交易信号
for i in range(1, 2):
    for j in range(30, 31):
        # ===策略参数
        para_list = [i, j]
        print('ma_short=%d, ma_long=%d' % (i, j))
        ma_short = para_list[0]  # 短期均线。ma代表：moving_average
        ma_long = para_list[1]  # 长期均线

        # ===计算均线。所有的指标，都要使用复权价格进行计算。
        df['ma_short'] = df['收盘价_复权'].rolling(ma_short, min_periods=1).mean()
        df['ma_long'] = df['收盘价_复权'].rolling(ma_long, min_periods=1).mean()

        # ===找出做多信号
        condition1 = df['ma_short'] > df['ma_long']  # 短期均线 > 长期均线
        condition2 = df['ma_short'].shift(1) <= df['ma_long'].shift(1)  # 上一周期的短期均线 <= 长期均线
        df.loc[condition1 & condition2, 'signal'] = 1  # 将产生做多信号的那根K线的signal设置为1，1代表做多

        # ===找出做多平仓信号
        condition1 = df['ma_short'] < df['ma_long']  # 短期均线 < 长期均线
        condition2 = df['ma_short'].shift(1) >= df['ma_long'].shift(1)  # 上一周期的短期均线 >= 长期均线
        df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

        # ===删除无关中间变量
        # df.drop(['ma_short', 'ma_long'], axis=1, inplace=True)

        # ===由signal计算出实际的每天持有仓位
        # 在产生signal的k线结束的时候，进行买入
        df['signal'].fillna(method='ffill', inplace=True)
        df['signal'].fillna(value=0, inplace=True)  # 将初始行数的signal补全为0
        df['pos'] = df['signal'].shift()
        df['pos'].fillna(value=0, inplace=True)  # 将初始行数的pos补全为0

        # ===删除无关中间变量
        df.drop(['signal'], axis=1, inplace=True)


        # 截图2007年之后的数据
        # df = df[df['交易日期'] >= pd.to_datetime('20070101')]


        # =====找出开仓、平仓条件
        condition1 = df['pos'] != 0
        condition2 = df['pos'] != df['pos'].shift(1)
        open_pos_condition = condition1 & condition2

        condition1 = df['pos'] != 0
        condition2 = df['pos'] != df['pos'].shift(-1)
        close_pos_condition = condition1 & condition2


        # =====对每次交易进行分组
        df.loc[open_pos_condition, 'start_time'] = df['candle_end_time']
        df['start_time'].fillna(method='ffill', inplace=True)
        df.loc[df['pos'] == 0, 'start_time'] = pd.NaT
        # print(df)
        # exit()

        # =====开始计算资金曲线
        # ===基本参数
        initial_cash = 1e9  # 初始资金，默认为1000000元
        slippage = 0.001  # 滑点，股票默认为0.01元，etf为0.001元
        c_rate = 1.0 / 10000  # 手续费，commission fees，默认为万分之2.5
        t_rate = 0.0 / 1000  # 印花税，tax，默认为千分之1,ETF没有印花税

        # ===在买入的K线
        # 在发出信号的当根K线以收盘价买入,每次都是买一百万
        df.loc[open_pos_condition, 'stock_num'] = initial_cash * (1 - c_rate) / (df['前收盘价'] + slippage)

        # 实际买入股票数量
        df['stock_num'] = np.floor(df['stock_num'] / 100) * 100

        # 买入股票之后剩余的钱，扣除了手续费
        df['cash'] = initial_cash - df['stock_num'] * (df['前收盘价'] + slippage) * (1 + c_rate)

        # 收盘时的股票净值
        df['stock_value'] = df['stock_num'] * df['收盘价']

        # ===在买入之后的K线
        # 买入之后现金不再发生变动
        df['cash'].fillna(method='ffill', inplace=True)
        df.loc[df['pos'] == 0, ['cash']] = None

        # 股票净值随着涨跌幅波动
        group_num = len(df.groupby('start_time'))
        if group_num > 1:
            t = df.groupby('start_time').apply(lambda x: x['收盘价_复权'] / x.iloc[0]['收盘价_复权'] * x.iloc[0]['stock_value'])
            t = t.reset_index(level=[0])
            df['stock_value'] = t['收盘价_复权']
        elif group_num == 1:
            t = df.groupby('start_time')[['收盘价_复权', 'stock_value']].apply(lambda x: x['收盘价_复权'] / x.iloc[0]['收盘价_复权'] * x.iloc[0]['stock_value'])
            df['stock_value'] = t.T.iloc[:, 0]

        # ===在卖出的K线
        # 股票数量变动
        df.loc[close_pos_condition, 'stock_num'] = df['stock_value'] / df['收盘价']

        # 现金变动
        df.loc[close_pos_condition, 'cash'] += df.loc[close_pos_condition, 'stock_num'] * (df['收盘价'] - slippage) * (1 - c_rate - t_rate)

        # 股票价值变动
        df.loc[close_pos_condition, 'stock_value'] = 0

        # ===账户净值
        df['net_value'] = df['stock_value'] + df['cash']

        # ===计算资金曲线
        df['equity_change'] = df['net_value'].pct_change(fill_method=None)  # 净资产涨跌幅
        df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'net_value'] / initial_cash - 1  # 开仓日的收益率
        df['equity_change'].fillna(value=0, inplace=True)
        df['equity_curve'] = (1 + df['equity_change']).cumprod()
        # print(df['equity_curve'])
        # exit()

        # ===删除无关数据
        df.drop(['start_time', 'stock_num', 'cash', 'stock_value', 'net_value'], axis=1, inplace=True)
        res = evaluate_investment(df, 'equity_curve', time='candle_end_time')
        print(res)
        plt.plot(df['candle_end_time'], df['收盘价'], label='策略')
        plt.plot(df['candle_end_time'], df['equity_curve'], label='择时后')
        plt.plot(df['candle_end_time'], df['ma_long'], label='MA20')
        plt.show()
exit()
# print(df)