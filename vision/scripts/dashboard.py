import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import streamlit as st
from dbscan_lib import dbscanAlgo as dbs

def dash():
    """ image = Image.open('../report/clusters.png')
    st.image(image, caption = 'Clusters')
     """
     
    tab1, tab2, tab3, tab4 = st.tabs(["Clustering", "Gaze Points", "Heatmap", "Overlay"])
    
    with tab1:
        st.header("Clusters")
    
        with st.container():
            st.image("../report/clusters.png")
    
    with tab2:
        st.header("Gaze Points")
    
        with st.container():
            st.image("../report/gaze_points.png")   
            
    with tab3:
        st.header("Heatmap")
    
        with st.container():
            st.image("../report/heatmap.png")   
    
    with tab4:
        st.header("Overlay Image")
    
        with st.container():
            st.image("../report/overlay_image.png")   
         
    
if __name__ == '__main__':
    dash()