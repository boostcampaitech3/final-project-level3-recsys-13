import streamlit as st

import io
import os
import yaml
import random
import string

from confirm_button_hack import cache_on_button_press
# import torch
# from model.args import parse_args
# from model.dataloader import Preprocess
# from model.model import LSTM, LSTMATTN, Bert
# from model import trainer
import requests
# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

def main():
    if "response" in st.session_state:
        st.title("Top10 레시피")
        rr = st.session_state.response["lists"]
        idx =  st.radio("",range(len(rr)), format_func=lambda x: rr[x]["name"])
        st.write(rr[idx]["description"])

        rate = st.text_input(label="평점")
        if st.button("추천받기"):
            try:
                recommended = requests.post("http://localhost:30002/api/v1/recommend", json={"rate":float(rate)}).json()
                st.write(recommended)
            except:
                st.write("실수가 아닙니다.")
        

def authenticate(id):
    return requests.post("http://localhost:30002/api/v1/login", json={"userid":id}).json()

def logined():
    if st.button("확인", key="confirm_button"):
        st.session_state.response = authenticate(id)
    main()

id = st.text_input('id')

logined()