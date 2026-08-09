import streamlit as st

# 1. Set up the Webpage layout
st.set_page_config(page_title="Aether-XAI", layout="wide")
st.title("Aether-XAI: ISRO Heavy Rain Predictor")
st.markdown("Upload a satellite image to see the weather prediction and the XAI Heatmap.")

# 2. Create an upload button
uploaded_file = st.file_uploader("Upload INSAT-3D Satellite Image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # 3. Create two columns for a split-screen look
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Satellite Image")
        st.image(uploaded_file, use_column_width=True)
        
    with col2:
        st.header("Grad-CAM XAI Analysis")
        st.warning("Prediction: 88% Chance of Heavy Rain")
        st.info("The AI model and Heatmap generator will be connected here soon!")