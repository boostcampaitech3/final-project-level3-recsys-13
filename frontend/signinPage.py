import streamlit as st
import requests
from config import API_DEST

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

def authenticate(name, password):
    # TODO : url 변경
    # 성공 시 10개 추천 레시피 받기
    # 실패 시 오류 메시지 출력
    cansignin = requests.post(f"{API_DEST}/api/v1/signin", json={"name":name, "password":password}).json()
    if cansignin["state"]=="Approved":
        # return {"lists":[{"name":i, "description":str(i)} for i in range(10)]}
        return [int(cansignin["user_id"]),cansignin["is_cold"]]
    elif cansignin["detail"]=="wrong password":
        st.write("잘못된 password 입니다.")
    else:
        st.write("잘못된 형식입니다.")
    return -1

def login():
    if "response" not in st.session_state:
        st.session_state.response=""

    if "id" not in st.session_state:
        st.session_state.id = ""

    if "password" not in st.session_state:
        st.session_state.password = ""

    if st.button("sign in", key="confirm_button"):
        st.session_state.user_id, st.session_state.cold = authenticate(st.session_state.id, st.session_state.password)
    if st.button("sign up", key = "signup_button"):
        st.session_state.page = "signupPage"
        st.experimental_rerun()

    if st.session_state.user_id != -1:
        if st.session_state.cold == True:
            st.session_state.page = "recThemePage"
            st.experimental_rerun()
        else:
            st.session_state.page = "recommendPage"
            st.experimental_rerun()

def signinPage():
    st.title("Login - CD test !")
    st.session_state.user_id = -1
    st.title(st.session_state.user_id)
    st.session_state.id = st.text_input('Id')
    st.session_state.password = st.text_input('Password')

    login()