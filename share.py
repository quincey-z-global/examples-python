import akshare as ak


# MODE = 'indices'
# MODE = 'index_stocks'
# MODE = 'index_daily'
# MODE = 'index_fund_daily'
MODE = 'stock_daily'
print()


# get the information of indices
# columns: ['index_code', 'display_name', 'publish_date']
if MODE == 'indices':
    df_indices = ak.index_stock_info()
    print(f'available indices:', '\n')
    print(df_indices, '\n')


# get the stock information of an index
# columns: ['品种代码', '品种名称', '纳入日期']
if MODE == 'index_stocks':
    index_code = '930050'
    df_index_stocks = ak.index_stock_cons(symbol=index_code)
    print(f'stocks in the index {index_code}:', '\n')
    print(df_index_stocks, '\n')


# get the historical daily values of an index
# columns: ['date', 'open', 'high', 'low', 'close', 'volume']
if MODE == 'index_daily':
    index_code = 'sh000015'
    df_index_daily = ak.stock_zh_index_daily(symbol=index_code)
    print(f'historical daily values of index {index_code}:', '\n')
    print(df_index_daily, '\n')


# get the historical daily values of an index fund
# column: ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
if MODE == 'index_fund_daily':
    index_fund_code = '510880'
    start_date, end_date = '20231207', '20250331'
    df_index_fund_daily = ak.fund_etf_hist_em(
        symbol=index_fund_code, period='daily', start_date=start_date, end_date=end_date, adjust='')
    print(f'historical daily values of index fund {index_fund_code}:', '\n')
    print(df_index_fund_daily, '\n')


# get the historical daily values of a stock
# columns: ['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
if MODE == 'stock_daily':
    stock_code = '000001'
    start_date, end_date = '20231207', '20251125'
    df_stock_daily = ak.stock_zh_a_hist(
        symbol=stock_code, period='daily', start_date=start_date, end_date=end_date, adjust='')
    print(f'historical daily values of stock {stock_code}:', '\n')
    print(df_stock_daily, '\n')
