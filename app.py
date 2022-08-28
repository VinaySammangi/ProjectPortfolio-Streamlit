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
from StockpriceForecasting.api_stockprice import *
from RecommendationEngine.content_based import *
import time
import pickle
import nltk
from textblob import TextBlob
import spacy
from spacy import displacy
import en_core_web_sm
#new packages
from transformers import pipeline
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration
from streamlit_chat import message as st_message

nltk.download('punkt')
nlp = en_core_web_sm.load()

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
HTML_WRAPPER_textsum = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; color:green">{}</div>"""

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


st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)
            
st.markdown("<i><text style='text-align: left; color:orange;'> Please view this website on either desktop or laptop</text></i>",unsafe_allow_html=True)
st.markdown("<h1 width: fit-content; style='text-align: center; color: white; background-color:#0083b8;'>Project Portfolio - Vinay Sammangi</h1>", unsafe_allow_html=True)        
set_page_container_style()

def _stockpriceforecasting():
    st1_11, st1_12, st1_13 = st.columns([2,12,1])
    st1_12 = st1_12.empty()
    st1_13 = st1_13.empty()
    st1_21, st1_23 = st.columns([1,1],gap="small")
    st1_31, st1_32, st1_33 = st.columns([1,20,1])

    ticker_list = ["AAPL","IBM","NVDA","INTC","CSCO","MSFT","INFY","ORCL","CRM","UBER"]
    longNames = ["Apple Inc.","International Business Machines Corporation","NVIDIA Corporation","Intel Corporation","Cisco Systems, Inc.","Microsoft Corporation","Infosys Limited","Oracle Corporation","Salesforce, Inc.","Uber Technologies, Inc."]
    tickerSymbol = st1_11.selectbox('Stock ticker', ticker_list)
    
    tickerData = yf.Ticker(tickerSymbol)
    technical_response = json.loads(technical_forecast(tickerSymbol))
    temp_df = pd.DataFrame(technical_response["DataFrame"])
    temp_df["Datetime"] = pd.to_datetime(temp_df["Datetime"])
    #string_logo = '<img src=%s height="50" align="right">' % tickerData.info['logo_url']
    string_name = longNames[ticker_list.index(tickerSymbol)]#tickerData.info['longName']
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
    #fig.update_layout(plot_bgcolor="rgb(245,245,245)")
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
        st1_32.markdown("<i> <h5 style='text-align: center; color:gray;'> Forecasts are generated between 10:00 and 15:30 EST during the market open days </h5> </i>",unsafe_allow_html=True)
    

@st.cache(allow_output_mutation=True)
def _load_netflix_movies():
    return pd.read_csv("RecommendationEngine/movies_data.csv")

def _netflix_recommendationengine():
    # Data Loading
    movies_data = _load_netflix_movies()
    # Header contents
    # st.markdown("<h2 width: fit-content; style='text-align: center; color: black;'>Netflix Movie Recommendation System</h2>", unsafe_allow_html=True)        

    st3_11, st3_12, st3_13, st3_14 = st.columns([3,2,5,1])
    
    # User-based preferences
    movie = st3_11.selectbox('Enter your favorite movie',movies_data["title"],)
    n_movies = st3_12.select_slider('Movies to recommend?',options=range(1,11))
    selected_variables = st3_13.multiselect('Variables to consider', ['title','director','cast','listed_in','country'],default=['title','director','cast','listed_in','country'])
    st3_21, st3_22 = st.columns([2,2])                
   
    # Perform top-10 movie recommendation generation
    if st3_14.button("Recommend"):
        top_recommendations = content_model(movie=movie,top_n = n_movies,features=selected_variables,movies_data=movies_data)
        st3_22.markdown("<h3 width: fit-content; style='text-align: left; color: gray;'>We think you'll like the below movie(s):</h3>", unsafe_allow_html=True)
        for i,j in enumerate(top_recommendations):
            st3_22.markdown(str(i+1)+'. '+j)
    st3_21.image('imgs/netflix_recommendation.png',use_column_width=True)


def _sentimentanalysis():
    def get_sentiment(value):
        if value > 0:
            return("😃 Positive")
        elif value < 0:
            return("😏 Negative")
        else:
            return("😐 Neutral")
    st.caption("""
        Sentiment analysis is a natural language processing (NLP) technique 
        used to determine whether text is positive, negative or neutral. Sentiment analysis is also known as “opinion mining” or “emotion artificial intelligence”. We can use it in e-commerce, politics, marketing, advertising, market research for example.
        """)
    text = st.text_input("Input text", "This movie is awesome. The acting is really bad though.")
    blob = TextBlob(text)
    
    st.write("Sentiment:",get_sentiment(blob.sentiment.polarity))
    if blob.sentiment.subjectivity > 0.5:
        st.write("Personal opinion: ✅")
    else:
        st.write("Personal opinion: ❌")
    
    st.write(blob.sentiment)

def analyze_text(text):
	return nlp(text)

def _nerproject():
    raw_text = st.text_area("Input Here","Ahead of India's much-anticipated Asia Cup 2022 opener against Pakistan, head coach Rahul Dravid is set to join the squad. The former India captain, who is expected to reach Dubai late on Saturday (August 27) night, will be in the dressing room during the clash against the arch-rivals. Cricbuzz also understands that VVS Laxman, who was named interim coach in the absence of Dravid, will fly back home. It is learnt that his return flight is on Saturday night itself and he will not be with the side for their tournament opener on Sunday. He is returning home tonight, said a source in Dubai.",height=30)
    if st.button("Analyze"):
        docx = analyze_text(raw_text)
        html = displacy.render(docx,style="ent")
        html = html.replace("\n\n","\n")
        st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_summarizer():
    model = pipeline("summarization")
    return model

def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')
    
    sentences = inp_str.split('<eos>')
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks

def _textsummarizer():
    summarizer = load_summarizer()
    st3_31, st3_32, st3_33 = st.columns([1,1,2]) 
    min = st3_31.slider('Minimum words in the summary', 10, 450, step=10, value=50)
    max = st3_32.slider('Maximum words in the summary', 50, 500, step=10, value=150)
    sentence = st.text_area('Please paste your article :',"Ahead of India's much-anticipated Asia Cup 2022 opener against Pakistan, head coach Rahul Dravid is set to join the squad. The former India captain, who is expected to reach Dubai late on Saturday (August 27) night, will be in the dressing room during the clash against the arch-rivals. Cricbuzz also understands that VVS Laxman, who was named interim coach in the absence of Dravid, will fly back home. It is learnt that his return flight is on Saturday night itself and he will not be with the side for their tournament opener on Sunday. He is returning home tonight, said a source in Dubai. Laxman's presence in Dubai was necessitated as Dravid, who tested positive for COVID-19, could not travel with the Indian side on August 23. Mr. Laxman has linked up with the squad in Dubai along with vice-captain Mr. KL Rahul, Mr. Deepak Hooda and Mr. Avesh Khan, who travelled from Harare, the Board of Control for Cricket in India (BCCI) had said on Laxman's arrival in Dubai from Harare while naming him the interim coach.",height=30)
    button = st.button("Summarize")    
    with st.spinner("Generating Summary.."):
        if button and sentence:
            chunks = generate_chunks(sentence)
            res = summarizer(chunks,
                              max_length=max, 
                              min_length=min
                              )
            text = ' '.join([summ['summary_text'] for summ in res])
            st.write(HTML_WRAPPER_textsum.format(text),unsafe_allow_html=True)            


@st.experimental_singleton
def get_chat_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    print(00)
    model_name = "facebook/blenderbot-400M-distill"
    print(0)
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    print(1)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    print(2)
    return tokenizer, model


def _chatbot():
    print(00000)
    # st3_41, st3_42 = st.columns([9,1]) 
    def generate_answer():
        print("chat models start")
        tokenizer, model = get_chat_models()
        print("chat models end")
        user_message = st.session_state.input_text
        print(3)
        inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
        print(4)
        result = model.generate(**inputs)
        print(5)
        message_bot = tokenizer.decode(
            result[0], skip_special_tokens=True
        )  # .replace("<s>", "").replace("</s>", "")
        print(6)    
        st.session_state.chat_history.append({"message": user_message, "is_user": True})
        st.session_state.chat_history.append({"message": message_bot, "is_user": False})

    print(0000)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    print(000)

    st.text_input("Talk to the bot",key="input_text", on_change=generate_answer)
    # click_ = st3_42.button('Restart chat')    
    # if click_:
    #     st.session_state.history = []    
    i = 0
    for chat in st.session_state.chat_history[::-1]:
        st_message(**chat,key="chatbot_"+str(i))  # unpacking
        i+=1    
    
        
def _machinetranslation():
    pass

def _toxicityranking():
    pass
        
def _nlp_applications():
    with st.expander("1.1. Netflix Recommendation System"):
        st.caption("""
            Content based - uses TF-IDF to recommend netflix movies.
            """)        
        _netflix_recommendationengine()
    
    with st.expander("1.2. Sentiment Analysis"):
        _sentimentanalysis()
        
    with st.expander("1.3. Named Entity Recognition"):
        st.caption("""
                 <i>Named entity recognition (NER) - sometimes referred to as entity chunking, extraction, or identification — is the task of identifying and categorizing key information (entities) in text.</i>
        """,unsafe_allow_html=True)
        _nerproject()
        
    with st.expander("1.4. Text Summarizer"):
        st.caption("""
                 <i>Summarization is the task of producing a shorter version of a document while preserving its important information. Some models can extract text from the original input, while other models (used here) can generate entirely new text.</i>
        """,unsafe_allow_html=True)
        _textsummarizer()

    with st.expander("1.5. Neural Chatbot"):
        st.caption("""
                 <i>Streamlit-Chat is a simple component, which provides a chat-app like interface, which makes a chatbot deployed on Streamlit have a cool UI.</i>
        """,unsafe_allow_html=True)
        pass
        # _chatbot()
    
    with st.expander("1.6. Machine Translation"):
        st.markdown("""
                 <i>A Seq2Seq model with attention mechanism trained on the Cornell movie dialog corpus</i>
        """,unsafe_allow_html=True)
        _machinetranslation()

    with st.expander("1.7. Severity of Toxic Comments"):
        st.markdown("""
                 <i>An ensemble model comprising DistilBERT and Ridge to rank relative ratings of toxicity between comments</i>
        """,unsafe_allow_html=True)
        _toxicityranking()
    
        
tab1, tab2, tab3, tab4 = st.tabs(["1. NLP Applications","2. Stock Price Forecasting Tool","3. CV Applications","4. NASA - Geospatial Analysis"])

    
with tab1:
    _nlp_applications()
    
with tab2:
    pass
    # _stockpriceforecasting()

       