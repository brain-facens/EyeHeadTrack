#!/home/nata-brain/miniconda3/envs/eyegaze/bin/ python
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import streamlit as st
from dbscan_lib import dbscanAlgo as dbs

def showImage():
    image = cv2.imread('/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/test_images/grocery-412912_1920.jpg')
    dsize = (1920, 1080)
    image = cv2.resize(image, dsize, interpolation =  cv2.INTER_LINEAR)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dash():
    """ image = Image.open('../report/clusters.png')
    st.image(image, caption = 'Clusters')
     """
     
    tab1, tab2, tab3, tab4 = st.tabs(["Clustering", "Gaze Points", "Heatmap", "Overlay"])
    
    # To cluster
    with tab1:
        st.header("Clusters")
    
        with st.container():
            st.image("../report/clusters.png")
            with st.expander("Entendea o gráfico"):
                st.write("Estes são os principais pontos onde o usuário manteve o seu olhar fixo.")
    
    # To gaze points
    with tab2:
        st.header("Gaze Points")
    
        with st.container():
            st.image("../report/gaze_points.png")   
            with st.expander("Entendea o gráfico"):
                st.write("Está é a trajetória que o usuário fez ao olhar para a imagem.")
                
    # To heatmaps        
    with tab3:
        st.header("Heatmap")
    
        with st.container():
            st.image("../report/heatmap.png")  
            with st.expander("Entendea o gráfico"):
                st.write("A coloração mais intensa, representa o ponto de maior fixação que o usário manteve o seu foco.") 
    
    # To Overlays Images
    with tab4:
        st.header("Overlay Image")
    
        with st.container():
            st.image("../report/overlay_image.png")   
            with st.expander("Entendea o gráfico"):
                st.write("Esta é a representação gráfica da sobreposição entre o mapa de calor e a imagem que o usuário viu.")
         
    
if __name__ == '__main__':
    dash()