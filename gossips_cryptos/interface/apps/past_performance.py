from pathlib import PureWindowsPath
import streamlit as st

import numpy as np
import pandas as pd
from PIL import Image
import datetime
#!/usr/bin/env python
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:

    import json



def plot_bitcoin_change(df):
    st.markdown("""### Bitcoin Price """)
    st.write('Here you can see the change of bitcoin price against of that our model predicted!')
    st.write('(actually now is open against close but soon you will be able to)')
    st.line_chart(df)

def get_past_data(start, end):
    #obtain past csv data
    past_performance = pd.read_csv('cryptocurrency_trading/data/4-month-BTC-perf.csv')
    past_performance['date'] = pd.to_datetime(past_performance['date'])
    past_performance.sort_values(by="date", inplace = True)
    past_performance['date'] = past_performance['date'].dt.date
    past_performance.set_index('date', inplace = True)
    df = past_performance[past_performance.index >= start]
    df = df[df.index <= end].copy()
    return df

def get_earnings(investment, earnings,start):

    ##### will not need this once model works:
    earnings.columns = ['pred', 'real']
    earnings['pred'] = earnings[['real']].pct_change()#.dropna(axis = 1)


    #############################################     KEEP THIS     ###########################################
    earnings['buy_sell'] = earnings['pred']>0
    earnings['pct'] = earnings['pred']+1
    earnings.dropna()
    ac_ = []
    for inx, row in earnings.iterrows():
        if row['buy_sell'] == True:
            ac_.append(row['pct'])
        else:
            ac_.append(1)
    earnings['prepare'] = ac_
    earnings['market'] = earnings['pct'].cumprod()
    earnings['acc_return'] = earnings['prepare'].cumprod()
    st.markdown("""### Making money!""")
    st.write('These are your earnings if you used our model against just investing and holding!!!')
    plot_ = earnings[['market', 'acc_return']]
    plot_.columns = ['Invest and hold', 'Return with our model']
    st.line_chart(plot_)
    our_return = earnings[earnings['buy_sell']]['pct'].prod()
    market_return = earnings['pct'].prod()
    extra = ((our_return/100)+1)*investment #- ((market_return/100)+1)*investment
    profit_model = extra - investment
    profit_market = ((market_return/100)+1)*investment - investment

    st.write("""
        <style>
        .big-font {
            font-size:50px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.write(f'If you used our model you will get a profit of ${round(profit_model,2)}, instead of just holding it, which would have given you a profit of ${round(profit_market,2)}')
    #st.write(f'If you used our model you will get a return of {round(our_return,2)}%, instead of just holding it, which would have given you a return of {round(market_return,2)}%')
    st.write('You now have:')
    st.write(f'<p class="big-font">$         {round(extra,2)}            </p>', unsafe_allow_html=True)
    st.write("Looks like you should've invested more money! 😉")



def app():
    st.markdown("""# Past performance """)

    st.write(' ')
    st.write('In this section you will be able to see how much money you would have made if you used our model in the past instead of just saving it!')

    # Ask for innitial investment:
    st.markdown("""### Let's Invest some money!""")
    investment = st.slider('How much money are we investing?', 100, 5000, 100)
    st.write(' ')
    st.write('Select a time period to invest!')
    min_start = datetime.datetime(2021, 5, 3)
    max_value = datetime.datetime(2021, 9, 7)
    start_date = st.date_input('Start date', min_start, min_value = min_start, max_value = max_value)
    end_date = st.date_input('End date', max_value, min_value  = min_start+datetime.timedelta(days=1), max_value = max_value)
    

    if start_date < end_date:
        past_performance = get_past_data(start_date, end_date)
        #plot_bitcoin_change(past_performance)
        get_earnings(investment, past_performance, start_date)
    else:
        st.error('Error: End date must fall after start date.')



## Exlain buy and sell everyday 