import streamlit as st
import requests

def signupPage():
    st.title("회원가입")
    name = st.text_input('Id')
    password1 = st.text_input("password", type="password")
    password2 = st.text_input("password 확인", type="password")
    if st.button("회원가입"):
        if password1!=password2:
            st.write("비밀번호가 일치하지 않습니다.")
        else:
            #TODO : url변경
            canSignup = requests.post("http://localhost:30002/signup", json={"name":name, "password":password1}).json()

            if canSignup["state"]=="Approved":
                st.session_state.page = "recommendPage"
                st.experimental_rerun()
            elif canSignup["detail"]=="duplicate error":
                st.write("중복된 아이디입니다.")
            else:
                st.write("잘못된 형식입니다.")

