#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 19:33:59 2022

@author: vinaysammangi
"""

import streamlit as st
from functools import reduce
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from st_aggrid import GridOptionsBuilder, AgGrid
from api_stockprice import *

BACKGROUND_COLOR = 'white'
COLOR = 'black'

def v_spacer(height,elem):
    for _ in range(height):
        elem.write('\n')
        
def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = True,
        padding_top: int = 0, padding_right: int = 0.5, padding_left: int = 0.5, padding_bottom: int = 0,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {padding_top}rem;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

im = Image.open("imgs/page_image.png")
st.set_page_config(page_title="Vinay Sammangi - Project Portfolio",page_icon=im,layout="wide")

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 1rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.markdown("<h1 width: fit-content; style='text-align: center; color: white; background-color:LightSkyBlue;'>Project Portfolio - Vinay Sammangi</h1>", unsafe_allow_html=True)        

st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True) 

set_page_container_style()

def _stockpriceforecasting():
    st1_11, st1_12, st1_13 = st.columns([2,12,1])
    st1_12 = st1_12.empty()
    st1_13 = st1_13.empty()
    st1_21, st1_23 = st.columns([1,1],gap="small")
    st1_31, st1_32, st1_33 = st.columns([1,20,1])
    
    ticker_list = ["AAPL","IBM","NVDA","INTC","CSCO","MSFT","INFY","ORCL","CRM","UBER"]
    tickerSymbol = st1_11.selectbox('Stock ticker', ticker_list)
    
    tickerData = yf.Ticker(tickerSymbol)
    technical_response = json.loads(technical_forecast(tickerSymbol))
    temp_df = pd.DataFrame(technical_response["DataFrame"])
    temp_df["Datetime"] = pd.to_datetime(temp_df["Datetime"])
    string_logo = '<img src=%s height="50" align="right">' % tickerData.info['logo_url']
    string_name = tickerData.info['longName']
    forecast_header1 = st1_12.empty()
    forecast_header1.markdown("<h2 style='text-align: center; color: gray;'>"+string_name+"</h2",unsafe_allow_html=True)
    #st1_13.markdown(string_logo,unsafe_allow_html=True)
    
    candlestick = go.Candlestick(x = temp_df['Datetime'], open=temp_df['Open'], high=temp_df['High'], low=temp_df['Low'], close=temp_df['Close'])
    max_date_plot = pd.to_datetime(str(max(temp_df["Datetime"])+pd.to_timedelta(1, unit='d'))[:10])
    min_date_plot = pd.to_datetime(str(min(temp_df["Datetime"]))[:10])

    last_hour = temp_df['Datetime'].dt.hour.iloc[-1]
    last_minute = temp_df['Datetime'].dt.minute.iloc[-1]
    
    fig = go.Figure()
    string_name = "Technical Chart"
    dir_color = "black"
    direction = "Wait till next day"
    # if (last_hour>=0) or (last_hour==15 and last_minute==0):
    if (last_hour>9 and last_hour < 15) or (last_hour==15 and last_minute==0):
        forecast_ = technical_response["Forecast"]
        forecast_time = temp_df['Datetime'].iloc[-1]+pd.to_timedelta(30, unit='m')
        last_closevalue = temp_df['Close'].iloc[-1]
        if last_closevalue<forecast_:
            direction = "BULLISH"
            dir_color = "green"
            string_name = "Technical Forecast: "+direction + " ("+str(technical_response["ForecastConfidence"])+"%)"
            fig.add_hline(y=last_closevalue,line_dash="dot",line_width=3,annotation_font_size=20,line_color="gray",annotation_text="Current Close",annotation_position="bottom",annotation_font_color="gray")
            fig.add_hline(y=forecast_,line_dash="dot",line_width=3,annotation_font_size=20,line_color="darkgreen",annotation_text="Forecast",annotation_position="top",annotation_font_color="green")
            st1_21.markdown("<h3 style='text-align: center; color: #06c729;'>"+string_name+"</h3",unsafe_allow_html=True)
        else:
            direction = "BEARISH"
            dir_color = "red"
            string_name = "Technical Forecast: "+direction + " ("+str(technical_response["ForecastConfidence"])+"%)"
            fig.add_hline(y=last_closevalue,line_dash="dot",line_color="gray",annotation_text="Current Close",annotation_position="top right",annotation_font_color="gray")
            fig.add_hline(y=forecast_,line_dash="dot",line_color="red",annotation_text="Forecast",annotation_position="bottom right",annotation_font_color="red")
            st1_21.markdown("<h3 style='text-align: center; color: #f59862;'>"+string_name+"</h3",unsafe_allow_html=True)
    
    fig.add_trace(candlestick)
    fig.update_layout(
    xaxis=dict(rangeslider=dict(
        visible=False
    ),type="date"))
    fig.update_layout(margin=dict(l=5, r=0, b=0, t=0),template="plotly_white")
    dt_all = pd.date_range(start=temp_df['Datetime'].iloc[0],end=temp_df['Datetime'].iloc[-1])
    dt_obs = [d.strftime("%Y-%m-%d") for d in temp_df['Datetime']]
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
    fig.update_layout(plot_bgcolor="rgb(245,245,245)")
    st1_21.plotly_chart(fig,use_container_width=True)
    
    sentiment_response = json.loads(sentiment_forecast(tickerSymbol))
    tweets_data = pd.DataFrame(sentiment_response["DataFrame"])
    text_conf = round(sentiment_response["ForecastConfidence"])
    text_sentiment = sentiment_response["Forecast"]
    colors = []
    for x in tweets_data["Sentiment"]:
        if x=="Bullish":
            colors.append("#D6FFA9")
        elif x=="Bearish":
            colors.append("#FFCDB0")
        else:
            colors.append("lightgray")
    
    del tweets_data["Sentiment"]
    fig = go.Figure(data=[go.Table(
    columnwidth = [20,400],
    header=dict(values=list(tweets_data.columns),
                align='left'),
    cells=dict(values=tweets_data.transpose().values.tolist(),
               align='left',fill_color=[colors]))
        ])
    fig.update_layout(margin=dict(l=40, r=0, b=40, t=4))
    if (last_hour>9 and last_hour < 15) or (last_hour==15 and last_minute==0):    
        if text_sentiment == "Bullish":
            string_name = "Market Sentiment Forecast: BULLISH ("+str(text_conf)+"%)"
            st1_23.markdown("<h3 style='text-align: center; color: #06c729;'>"+string_name+"</h3",unsafe_allow_html=True)
        if text_sentiment == "Bearish":
            string_name = "Market Sentiment Forecast: BEARISH ("+str(text_conf)+"%)"
            st1_23.markdown("<h3 style='text-align: center; color: #f59862;'>"+string_name+"</h3",unsafe_allow_html=True)
        
    st1_23.plotly_chart(fig,use_container_width=True)

    if (last_hour>9 and last_hour < 15) or (last_hour==15 and last_minute==0):    
        final_forecast = "NEUTRAL"
        final_color = "lightgray"
        if text_sentiment.upper()==direction.upper():
            if text_sentiment=="Bullish":
                final_forecast = "BULLISH"
                final_color = "#06c729"
            elif text_sentiment=="Bearish":
                final_forecast = "BEARISH"
                final_color = "#f59862"

        string_name = tickerData.info['longName']    
        #forecast_header1.markdown("<h3 style='text-align: right; color: black;'>"+string_name+"Overall Forecast: </h3",unsafe_allow_html=True)
        forecast_header1.markdown("<h2 style='text-align: center; color: "+final_color+";'>"+string_name+" Overall Forecast: "+final_forecast+"</h2>",unsafe_allow_html=True)
    else:
        st1_32.markdown("<i> <h4 style='text-align: center; color:gray;'> Forecasts are generated between 10:00 and 15:30 EST during the market open days </h4> </i>",unsafe_allow_html=True)


def _nlp_applications():
    pass

def _netflix_recommendationengine():
    pass

tab1, tab2, tab3, tab4 = st.tabs(["1. Stock Price Forecasting Tool","2. NLP Applications","3. Netflix Recommendation System","4. NASA - Geospatial Analysis"])

with tab1:
    _stockpriceforecasting()
    
with tab2:
    _nlp_applications()
    
with tab3:
    _netflix_recommendationengine()

       