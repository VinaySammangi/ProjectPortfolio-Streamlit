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
    pass

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

       