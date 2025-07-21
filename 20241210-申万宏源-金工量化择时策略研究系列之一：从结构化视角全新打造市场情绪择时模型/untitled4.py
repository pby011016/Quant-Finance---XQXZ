import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# 设置Tushare token 
ts.set_token('4e250e621fb484356a7b948649e3f1c73d9af358e9ce6a91c1e68a77')
pro = ts.pro_api()

# 获取融资融券数据 (2017-01-03至今)
def get_margin_data():
    # 获取沪市融资数据
    sh_margin = pro.margin(start_date='20170103', end_date=pd.Timestamp.today().strftime('%Y%m%d'))
    sh_margin['trade_date'] = pd.to_datetime(sh_margin['trade_date'], format='%Y%m%d')
    sh_margin.set_index('trade_date', inplace=True)
    
    # 获取深市融资数据
    sz_margin = pro.sz_margin(start_date='20170103', end_date=pd.Timestamp.today().strftime('%Y%m%d'))
    sz_margin['trade_date'] = pd.to_datetime(sz_margin['trade_date'], format='%Y%m%d')
    sz_margin.set_index('trade_date', inplace=True)
    
    # 合并两市数据
    margin_data = pd.DataFrame()
    margin_data['total_balance'] = sh_margin['rzye'] + sz_margin['rzye']  # 两市融资余额总和
    
    return margin_data

# 获取沪深300指数数据
def get_hs300_data():
    hs300 = pro.index_daily(ts_code='000300.SH', start_date='20170103', 
                           end_date=pd.Timestamp.today().strftime('%Y%m%d'))
    hs300['trade_date'] = pd.to_datetime(hs300['trade_date'], format='%Y%m%d')
    hs300.set_index('trade_date', inplace=True)
    hs300 = hs300.sort_index()
    
    # 计算净值 (以2017-01-03为基准)
    base_value = hs300.loc['2017-01-03', 'close']
    hs300['nav'] = hs300['close'] / base_value
    
    return hs300[['nav']]

# 获取自由流通市值数据
def get_free_float_mv():
    # 获取全市场每日指标
    market_data = pro.daily_basic(start_date='20170103', end_date='20250601', 
                                 fields='trade_date,circ_mv')
    market_data['trade_date'] = pd.to_datetime(market_data['trade_date'], format='%Y%m%d')
    market_data.set_index('trade_date', inplace=True)
    market_data = market_data.sort_index()
    
    # 自由流通市值 = 全市场流通市值
    return market_data.rename(columns={'circ_mv': 'free_float_mv'})

# 主程序
def main():
    # 获取数据
    margin_data = get_margin_data()
    hs300 = get_hs300_data()
    market_value = get_free_float_mv()
    
    # 合并数据
    merged = pd.concat([margin_data, market_value, hs300], axis=1)
    merged = merged.dropna()
    
    # 计算融资余额占自由流通市值比例
    merged['ratio'] = merged['total_balance'] / merged['free_float_mv']
    
    # 计算60日均线
    merged['ratio_60ma'] = merged['ratio'].rolling(window=60).mean()
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制融资余额占比
    ax1 = plt.gca()
    line1, = ax1.plot(merged.index, merged['ratio_60ma'], color='#1f77b4', linewidth=2.5, label='融资余额占比(60MA)')
    ax1.set_ylabel('融资余额/自由流通市值', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_ylim(0.035, 0.048)
    
    # 绘制沪深300净值
    ax2 = ax1.twinx()
    line2, = ax2.plot(merged.index, merged['nav'], color='#ff7f0e', linewidth=2.5, label='沪深300净值')
    ax2.set_ylabel('沪深300净值(2017-01-03=1.0)', fontsize=12)
    ax2.tick_params(axis='y')
    ax2.set_ylim(0.6, 1.5)
    
    # 设置标题和网格
    plt.title('A股市场融资余额占自由流通市值比 (2017/1/3 - 最新)', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 设置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    
    # 添加图例
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=10)
    
    # 添加数据来源标注
    plt.figtext(0.75, 0.02, '数据来源: Tushare', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('融资余额占自由流通市值比.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()