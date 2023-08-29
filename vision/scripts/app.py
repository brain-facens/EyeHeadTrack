import streamlit as st
import dbscan_lib as dbs

data = dbs.dbscanAlgo()
data.dfHandling()

clusters    = data.plotClusters()
gaze_points = data.plotGazePoints()
density     = data.plotDensity()
overlay     = data.overlayImageData()
  
# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Selecione a sessão:",
    ("Natanael","Outro")
)

with st.sidebar:
    st.write('Análise dos testes realizados do o sistema de Gaze Tracking')


col1, col2 = st.columns(2)

with st.container():
    with col1:
        st.title("Cluster Points")
        st.pyplot(clusters)

    with col2:
        st.title("Gaze Points")
        st.pyplot(gaze_points) 
        
with st.container():
    with col1:
        st.title("Heatmap")
        st.pyplot(density)

    with col2:
        st.title("Overlay")
        st.pyplot(overlay) 