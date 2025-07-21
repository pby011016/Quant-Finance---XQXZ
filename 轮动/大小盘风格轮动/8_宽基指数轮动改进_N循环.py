'''
大小盘风格轮动
两个标的谁近二十天涨幅多，买谁
改进：二十天涨跌幅都小于0，则空仓
遍历找最优动量参数
'''
import pandas as pd
import numpy as np
from function import *
import matplotlib.pyplot as plt

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数



# 设置参数
trade_rate = 0.6 / 10000  # 场内基金万分之0.6，买卖手续费相同，无印花税
N = range(1, 60)  # 计算多少天的动量
# print(N)
# exit()
for momentum_days in N:
    # 读取数据
    # 大盘： 上证50（000016）/沪深300（000300）/中证500（000905）
    # 小盘： 中证500（000905）/中证1000（000852）/创业板（399006）/科创50（000688）
    # df = pd.DataFrame()
    df_50 = pd.read_csv('sh000016.csv', encoding='gbk', parse_dates=['candle_end_time'])
    df_300 = pd.read_csv('sh000300.csv', encoding='gbk', parse_dates=['candle_end_time'])
    df_500 = pd.read_csv('sh000905.csv', encoding='gbk', parse_dates=['candle_end_time'])
    df_1000 = pd.read_csv('sh000852.csv', encoding='gbk', parse_dates=['candle_end_time'])
    df_chuang = pd.read_csv('sz399006.csv', encoding='gbk', parse_dates=['candle_end_time'])
    df_kc50 = pd.read_csv('sh000688.csv', encoding='gbk', parse_dates=['candle_end_time'])
    # 计算大小盘每天的涨跌幅amplitude
    df_50['50_amp'] = df_50['close'] / df_50['close'].shift(1) - 1
    df_300['300_amp'] = df_300['close'] / df_300['close'].shift(1) - 1
    df_500['500_amp'] = df_500['close'] / df_500['close'].shift(1) - 1
    df_1000['1000_amp'] = df_1000['close'] / df_1000['close'].shift(1) - 1
    df_chuang['chuang_amp'] = df_chuang['close'] / df_chuang['close'].shift(1) - 1
    df_kc50['kc50_amp'] = df_kc50['close'] / df_kc50['close'].shift(1) - 1
    # 重命名行
    df_50.rename(columns={'open': '50_open', 'close': '50_close'}, inplace=True)
    df_300.rename(columns={'open': '300_open', 'close': '300_close'}, inplace=True)
    df_500.rename(columns={'open': '500_open', 'close': '500_close'}, inplace=True)
    df_1000.rename(columns={'open': '1000_open', 'close': '1000_close'}, inplace=True)
    df_chuang.rename(columns={'open': 'chuang_open', 'close': 'chuang_close'}, inplace=True)
    df_kc50.rename(columns={'open': 'kc50_open', 'close': 'kc50_close'}, inplace=True)
    # 合并数据
    df = pd.merge(left=df_50[['candle_end_time', '50_open', '50_close', '50_amp']], left_on=['candle_end_time'],
                  right=df_300[['candle_end_time', '300_open', '300_close', '300_amp']],
                  right_on=['candle_end_time'], how='left')
    df = pd.merge(left=df, left_on=['candle_end_time'],
                  right=df_500[['candle_end_time', '500_open', '500_close', '500_amp']], right_on=['candle_end_time'],
                  how='left')
    df = pd.merge(left=df, left_on=['candle_end_time'],
                  right=df_1000[['candle_end_time', '1000_open', '1000_close', '1000_amp']], right_on=['candle_end_time'],
                  how='left')
    df = pd.merge(left=df, left_on=['candle_end_time'],
                  right=df_chuang[['candle_end_time', 'chuang_open', 'chuang_close', 'chuang_amp']], right_on=['candle_end_time'],
                  how='left')
    # df = pd.merge(left=df, left_on=['candle_end_time'],
    #               right=df_kc50[['candle_end_time', 'kc50_open', 'kc50_close', 'kc50_amp']], right_on=['candle_end_time'],
    #               how='left')
    df = df.dropna()
    # print(df.head(5))
    # exit()
    # 计算N日的动量momentum (计算N日的涨跌幅)
    df['50_mom'] = df['50_close'].pct_change(periods=momentum_days)
    df['300_mom'] = df['300_close'].pct_change(periods=momentum_days)
    df['500_mom'] = df['500_close'].pct_change(periods=momentum_days)
    df['1000_mom'] = df['1000_close'].pct_change(periods=momentum_days)
    df['chuang_mom'] = df['chuang_close'].pct_change(periods=momentum_days)
    # df['kc50_mom'] = df['kc50_close'].pct_change(periods=momentum_days)
    # print(df.head(10))
    # exit()
    # 风格变换条件
    condition_50 = (df['50_mom'] > df['300_mom']) & (df['50_mom'] > df['500_mom']) & (df['50_mom'] > df['1000_mom']) & (df['50_mom'] > df['chuang_mom'])  # & (df['50_mom'] > df['kc50_mom'])
    condition_300 = (df['300_mom'] > df['50_mom']) & (df['300_mom'] > df['500_mom']) & (df['300_mom'] > df['1000_mom']) & (df['300_mom'] > df['chuang_mom'])  # & (df['300_mom'] > df['kc50_mom'])
    condition_500 = (df['500_mom'] > df['50_mom']) & (df['500_mom'] > df['300_mom']) & (df['500_mom'] > df['1000_mom']) & (df['500_mom'] > df['chuang_mom'])  # & (df['500_mom'] > df['kc50_mom'])
    condition_1000 = (df['1000_mom'] > df['50_mom']) & (df['1000_mom'] > df['300_mom']) & (df['1000_mom'] > df['500_mom']) & (df['1000_mom'] > df['chuang_mom'])  # & (df['1000_mom'] > df['kc50_mom'])
    condition_chuang = (df['chuang_mom'] > df['50_mom']) & (df['chuang_mom'] > df['300_mom']) & (df['chuang_mom'] > df['500_mom']) & (df['chuang_mom'] > df['1000_mom'])  # & (df['chuang_mom'] > df['kc50_mom'])
    # condition_kc50 = (df['kc50_mom'] > df['50_mom']) & (df['kc50_mom'] > df['300_mom']) & (df['kc50_mom'] > df['500_mom']) & (df['kc50_mom'] > df['1000_mom']) & (df['kc50_mom'] > df['chuang_mom'])
    condition_empty = (df['50_mom'] < 0) & (df['300_mom'] < 0) & (df['500_mom'] < 0) & (df['1000_mom'] < 0) & (df['chuang_mom'] < 0)  # & (df['kc50_mom'] < 0)
    df.loc[condition_300, 'style'] = '300'
    df.loc[condition_500, 'style'] = '500'
    df.loc[condition_1000, 'style'] = '1000'
    df.loc[condition_chuang, 'style'] = 'chuang'
    # df.loc[condition_kc50, 'style'] = 'kc50'
    df.loc[condition_empty, 'style'] = 'empty'

    # 相等时维持原来的仓位。
    df['style'].fillna(method='ffill', inplace=True)
    # 收盘才能确定风格，实际的持仓pos要晚一天。
    df['pos'] = df['style'].shift(1)
    # 删除持仓为nan的天数（创业板2010年才有）
    df.dropna(subset=['pos'], inplace=True)
    # 计算策略的整体涨跌幅strategy_amp
    df.loc[df['pos'] == '50', 'strategy_amp'] = df['50_amp']
    df.loc[df['pos'] == '300', 'strategy_amp'] = df['300_amp']
    df.loc[df['pos'] == '500', 'strategy_amp'] = df['500_amp']
    df.loc[df['pos'] == '1000', 'strategy_amp'] = df['1000_amp']
    df.loc[df['pos'] == 'chuang', 'strategy_amp'] = df['chuang_amp']
    # df.loc[df['pos'] == 'kc50', 'strategy_amp'] = df['kc50_amp']
    df.loc[df['pos'] == 'empty', 'strategy_amp'] = 0

    # 调仓时间
    df.loc[df['pos'] != df['pos'].shift(1), 'trade_time'] = df['candle_end_time']
    # 将调仓日的涨跌幅修正为开盘价买入涨跌幅（并算上交易费用，没有取整数100手，所以略有误差）
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == '50'), 'strategy_amp_adjust'] = df['50_close'] / (
            df['50_open'] * (1 + trade_rate)) - 1
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == '300'), 'strategy_amp_adjust'] = df['300_close'] / (
            df['300_open'] * (1 + trade_rate)) - 1
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == '500'), 'strategy_amp_adjust'] = df['500_close'] / (
            df['500_open'] * (1 + trade_rate)) - 1
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == '1000'), 'strategy_amp_adjust'] = df['1000_close'] / (
            df['1000_open'] * (1 + trade_rate)) - 1
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == 'chuang'), 'strategy_amp_adjust'] = df['chuang_close'] / (
            df['chuang_open'] * (1 + trade_rate)) - 1
    # df.loc[(df['trade_time'].notnull()) & (df['pos'] == 'kc50'), 'strategy_amp_adjust'] = df['kc50_close'] / (
    #         df['kc50_open'] * (1 + trade_rate)) - 1
    df.loc[df['trade_time'].isnull(), 'strategy_amp_adjust'] = df['strategy_amp']
    # 扣除卖出手续费
    df.loc[(df['trade_time'].shift(-1).notnull()), 'strategy_amp_adjust'] = (1 + df[
        'strategy_amp']) * (1 - trade_rate) - 1
    del df['strategy_amp'], df['style']

    df.reset_index(drop=True, inplace=True)
    # 计算净值
    df['50_net'] = df['50_close'] / df['50_close'][0]
    df['300_net'] = df['300_close'] / df['300_close'][0]
    df['500_net'] = df['500_close'] / df['500_close'][0]
    df['1000_net'] = df['1000_close'] / df['1000_close'][0]
    df['chuang_net'] = df['chuang_close'] / df['chuang_close'][0]
    # df['kc50_net'] = df['kc50_close'] / df['kc50_close'][0]
    df['strategy_net'] = (1 + df['strategy_amp_adjust']).cumprod()

    # 评估策略的好坏
    res = evaluate_investment(df, 'strategy_net', time='candle_end_time')
    print('计算 %d 日动量为时' % momentum_days)
    print(res)

# # 绘制图形
# plt.plot(df['candle_end_time'], df['strategy_net'], label='strategy')
# plt.plot(df['candle_end_time'], df['50_net'], label='50_net')
# plt.plot(df['candle_end_time'], df['300_net'], label='300_net')
# plt.plot(df['candle_end_time'], df['500_net'], label='500_net')
# plt.plot(df['candle_end_time'], df['1000_net'], label='1000_net')
# plt.plot(df['candle_end_time'], df['chuang_net'], label='chuang_net')
# # plt.plot(df['candle_end_time'], df['kc50_net'], label='kc50_net')
# plt.legend()
# plt.show()
#
# # 保存文件
# print(df.tail(10))
# df.to_csv('宽基指数轮动改进结果.csv', encoding='gbk', index=False)
