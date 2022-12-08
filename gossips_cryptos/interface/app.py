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
    # Fall back to Python 2's urllib2

    import json
from multiapp import MultiApp
from apps import home, past_performance, prediction # import your app modules here


st.set_page_config(
            page_title="Quick reference", # => Quick reference - Streamlit
            page_icon="🐍",
            layout="centered", # wide
            initial_sidebar_state="auto") # collapsed

# df = pd.DataFrame({
#     'first column': ['Some general info please' , 'Check out past performance', 'Give me tradding advice'],
#     'second column': [10, 20, 30]
# })
# option = st.sidebar.selectbox(
#     'What do you want to use our services for?',
#      df['first column'])

# 'You selected:', option

app = MultiApp()
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Predict", prediction.app)
app.add_app("Past Performance", past_performance.app)


# The main app
app.run()


# ### THIS WILL GO ON THE FIRST PAGe

# st.markdown("""# Cryptocurrency trading!
# ## We predict the movement in bitcoin price, taking into account sentiment analysis from reddit and twitter
# Do you wanna have a go yourself?""")

# def get_jsonparsed_data():
#     """
#     Receive the content of ``url``, parse it as JSON and return the object.
#     Parameters
#     ----------
#     url : str
#     Returns
#     -------
#     dict
#     """
#     url = ("https://financialmodelingprep.com/api/v3/quote/BTCUSD?apikey=a58413697e8263de9c95cab92049ea3f")
#     response = urlopen(url)
#     data = response.read().decode("utf-8")
#     return json.loads(data)

# # print(get_jsonparsed_data(url)[0]['price'])
# bitcoin_live_data = get_jsonparsed_data()[0]
# bitcoin_current_price = bitcoin_live_data ['price']
# bitcoin_change = bitcoin_live_data ['changesPercentage']


# # number = st.number_input('How much money are you investing???')

# # st.write('You are investing: ', number)


# col1, col2, col3 = st.columns(3)
# col1.metric("", "", "")
# col2.metric("BITCOIN", f"${bitcoin_current_price}", f"{bitcoin_change}%")
# col3.metric("", "", "")
# # perhaps insert here the current value of bitcoin?



# ### THIS WILL GO ON THE SECOND PAGE

# st.markdown("""# Past performance """)

# def plot_bitcoin_change(df):
#     st.markdown("""### Bitcoin Price """)
#     st.write('Here you can see the change of bitcoin price against of that our model predicted!')
#     st.write('(actually now is open against close but soon you will be able to)')
#     st.line_chart(df)

# def get_past_data(start, end):
#     #obtain past csv data
#     past_performance = pd.read_csv('../cryptocurrency_trading/data/4-month-BTC-perf.csv')
#     past_performance['date'] = pd.to_datetime(past_performance['date'])
#     past_performance.sort_values(by="date", inplace = True)
#     past_performance['date'] = past_performance['date'].dt.date
#     past_performance.set_index('date', inplace = True)
#     df = past_performance[past_performance.index >= start]
#     df = df[df.index <= end].copy()
#     return df

# def get_earnings(investment, earnings,start):
#     yesterday  = start - datetime.timedelta(days=1)
#     initial_investment = pd.DataFrame([investment], index = [yesterday], columns = ['dif'])
#     earnings.columns = ['pred', 'real']
#     earnings['dif'] = earnings['pred'] - earnings['real']
#     difference = initial_investment.append(earnings[['dif']])
#     sum = difference.cumsum()
#     sum.columns = ['Bitcoin earnings']
#     sum['No investment'] = investment
#     st.markdown("""### Making money!""")
#     st.write('These are your earnings if you used our model against just saving it!')
#     st.line_chart(sum)
#     last_money = sum.iloc[-1:,:]
#     extra_cash = last_money['Bitcoin earnings'] - last_money['No investment']
#     st.write(f"If you used our model instead of just saving it you woul've made an extra: ${extra_cash.values[0]}")


# st.write(' ')
# st.write('In this section you will be able to see how much money you would have made if you used our model in the past instead of just saving it!')

# # Ask for innitial investment:
# st.markdown("""### Let's Invest some money!""")
# investment = st.slider('How much money are we investing?', 100, 1000, 10)
# st.write(' ')
# st.write('Select a time period to invest!')
# min_start = datetime.datetime(2021, 5, 3)
# max_value = datetime.datetime(2021, 9, 7)
# start_date = st.date_input('Start date', min_start, min_value = min_start, max_value = max_value)
# end_date = st.date_input('End date', max_value, min_value  = min_start+datetime.timedelta(days=1), max_value = max_value)

# if start_date < end_date:
#     #st.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
#     past_performance = get_past_data(start_date, end_date)
#     plot_bitcoin_change(past_performance)
#     get_earnings(investment, past_performance, start_date)
# else:
#     st.error('Error: End date must fall after start date.')




# ### THIS WILL GO ON THE THIRD PAGE
# buy = 0


# st.markdown("""## Using the model """)

# if st.button('Show me what to do with my money!'):
#     # print is visible in the server output, not in the page
#     print('The model will tell you what to do')
#     # st.write('I was clicked 🎉')
#     # st.write('Further clicks are not visible but are executed')

#     if buy == 1:
#         image = Image.open('photos_frontend/Buy-Bitcoin.jpg')
#     else:
#         image = Image.open('sell-bitcoin.jpg')
#     st.image(image, caption='Our recommendation!!')
#     st.write('👎SELL!!👎')
#     st.write('👍BUY!!👍')




# else:
#     st.write('Will it be sell or buy!!')




# CSS = """
# h1 {
#     color: red;
# }

# """

# if st.checkbox('Inject CSS'):
#     st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

# df = pd.DataFrame({
#           'first column': list(range(1, 11)),
#           'second column': np.arange(10, 101, 10)
#         })

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
# line_count = st.slider('Select a line count', 1, 10, 3)

# # and used in order to select the displayed lines
# head_df = df.head(line_count)

# head_df
