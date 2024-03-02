import streamlit as st
import numpy as np
import pandas as pd
import pickle

df=pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))


st.sidebar.title("About")
st.sidebar.info(
    "This app is created to predict the price of laptops based on various configurations "
    "using the machine learning models. It aims to provide an easy and quick way to get an "
    "estimate for the price of a laptop with up to 80% accuracy."
)

st.title("Laptop Price Predictor")
st.markdown("#### Estimate the price of your desired laptop based on configuration with my ML-powered prediction tool.")
st.markdown("#### It's `Quick, accurate, and easy to use`.")

company=st.selectbox('Brand',df['Company'].unique())
type_name=st.selectbox('Type',df['TypeName'].unique())
ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,32,64])
weight=st.number_input('Weight of the Laptop')
touchscreen=st.selectbox('Touchscreen',['No','Yes'])
ips=st.selectbox('IPS_panel',['No','Yes'])
screen_size=st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu=st.selectbox('CPU brand',df['CPU brand'].unique())
hdd=st.selectbox('HDD',[0,128,256,512,1024,2048])
ssd=st.selectbox('SSD',[0,128,256,512,1024])
gpu=st.selectbox('Gpu brand',df['Gpu brand'].unique())
os=st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips=='Yes':
        ips=1
    else:
        ips=0

    X_res,Y_res=[int(x) for x in resolution.split('x')]
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size
   
    input_data = pd.DataFrame([{
    'Company': company,
    'TypeName': type_name,
    'Ram': ram,
    'Weight': weight,
    'Touchscreen': touchscreen,
    'IPS_panel': ips,
    'ppi': ppi,
    'CPU brand': cpu,
    'HDD': hdd,
    'SSD': ssd,
    'Gpu brand': gpu,
    'os': os
    }])
    st.title("Predicted Price is : "+str(int(pipe.predict(input_data)))+" Rs.")




    















