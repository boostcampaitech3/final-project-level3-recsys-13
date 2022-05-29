import streamlit as st
from signinPage import signinPage
from recommendPage import recommendPage
from signupPage import signupPage

if "page" not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page=="login":
    signinPage()
if st.session_state.page=="recommendPage":
    recommendPage()
if st.session_state.page=="signupPage":
    signupPage()