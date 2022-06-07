import streamlit as st
import requests
import numpy as np
import streamlit_modal as modal
from config import API_DEST


    
def setGrid():
    st.session_state.num_themes = requests.get(f"{API_DEST}/api/v1/num_themes").json()
    list_num = 6
    rec_num = 5
    samples = np.random.choice(range(st.session_state.num_themes['num']), list_num, replace=False)
    st.session_state.theme = [] 
    grid = [st.container()]*list_num
    for i in range(list_num):
        st.session_state.theme.append(requests.get(f"{API_DEST}/api/v1/theme/{samples[i]}").json()) 
        grid[i].title(st.session_state.theme[i]['theme_title'])
        grid[i].cols = grid[i].columns(rec_num)
        for j in range(rec_num):
            st.session_state.detail = st.session_state.theme[i]['samples'][j]['id']
            grid[i].cols[j].image('https://img.sndimg.com/food/image/upload/f_auto,c_thumb,q_73,ar_16:9,w_768/v1/img/recipes/56/3/x5DL56UORN2WSzPK6bEL_IMG_4001.JPG')
            grid[i].cols[j].write(st.session_state.theme[i]['samples'][j]['title'])
    return grid
    
def recThemePage():
    rf = st.button('refresh')
    if "theme" not in st.session_state:
        grid = setGrid()
    if rf:
        st.empty()
        grid = setGrid()
            
    # if open_modal:
    #     modal.open()
        
    # if modal.is_open():
    #     with modal.container():
    #         st.write(st.session_state.detail)