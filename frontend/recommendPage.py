import streamlit as st
import requests

ingredients = []
def recommendPage():
    if "recommend" not in st.session_state:
        st.session_state.recommend = ""
    
    st.title("추천 레시피 정보 입력")
    datum = st.multiselect("필요 정보를 입력해주세요", options = ["재료", "칼로리", "지방", "설탕", "나트륨", "단백질", "포화 지방", "탄수화물"
    , "제조 시간", "제조 과정 수"])
    if "재료" in datum:
        ingredients.append(
            st.multiselect("재료를 입력해주세요", options = ["green", "yellow"])
        )
    if "칼로리" in datum:
        calories = st.text_input("칼로리량")
    if "지방" in datum:
        totalfat = st.text_input("지방량")
    if "설탕"  in datum:
        suger = st.text_input("설탕량")
    if "나트륨" in datum:
        sodium = st.text_input("나트륨량")
    if "단백질" in datum:
        protein = st.text_input("단백질량")
    if "포화 지방" in datum:
        sturatedfat = st.text_input("포화 지방량")
    if "탄수화물" in datum:
        carbohydrates = st.text_input("탄수화물량")
    if "제조 시간" in datum:
        minutes = st.text_input("제조 시간")
    if "제조 과정 수" in datum:
        steps = st.slider("제조 과정 수", 0, 100, (0, 10))



    if st.button("추천"):
        st.session_state.recommend = 1
        # TODO : url 변경, 재료 입력받기 및 추천정보 받기 API 필요 
        # st.session_state.recipes = requests.post("http://localhost:30002/recten", json={"userid":int(rate)}).json()
        # 아래는 임시 recipes 정보입니다.
        st.session_state.recipes = {"lists":[{"name":i, "description":str(i)} for i in range(10)]}
    
    if st.session_state.recommend==1:
        st.title("Top10 레시피")
        rr = st.session_state.response["lists"]
        idx =  st.radio("",range(len(rr)), format_func=lambda x: rr[x]["name"])
        st.write(rr[idx]["description"])



        rate = st.text_input(label="평점")
        if st.button("평점 남기기"):
            # TODO : url 변경
            try:
                recommended = requests.post("http://localhost:30002/score", json={"rate":float(rate)}).json()
                recommended = rate
                st.write(recommended)
            except:
                st.write("실수가 아닙니다.")
    