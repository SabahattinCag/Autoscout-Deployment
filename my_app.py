import pandas as pd
import numpy as np
import pickle
import streamlit as st

st.markdown("""<p style="background-color:blue; color:floralwhite; font-size:300%; text-align:center; border-radius:10px 10px; font-family:newtimeroman; line-height: 1.4;">Auto Scout - Car Price Prediction</p>""", unsafe_allow_html=True)
st.markdown('Auto Scout data which using for this project, scraped from the on-line car trading company(https://www.autoscout24.com) in 2019, contains many features of 9 different car models. In this project, I use two machine learning algorithms, **Linear Regression** and **Random Forest**.')

df = pd.read_csv('final_scout_not_dummy.csv')
st.write(df.head(2))
st.markdown('I have determined that the following features have a high impact on the price. Therefore, we can use them in machine learning algorithms.\n "make_model", "hp_kW", "km","age", "Gearing_Type", "Drive_chain","Type", "price"')
df_new = df[["make_model", "hp_kW", "km","age", "Gearing_Type", "Drive_chain","Type", "price"]]
st.write(df_new.head(2))

st.sidebar.header('Select the machine learning algorithm')
model=st.sidebar.selectbox("Algorithm", ['Linear Regression','Random Forest regression'])


st.sidebar.header('Select the properties of the car')
brand_model = df_new.make_model.unique()
makemodel=st.sidebar.selectbox("Car brand and model (make_model)", brand_model)
hpower = st.sidebar.number_input("Horse power (hp_kW)", min_value=40,max_value=300)
km = st.sidebar.number_input("km", min_value=0,max_value=320000)
age = st.sidebar.slider("Age", min_value=0, max_value=5, value=1, step=1)
transmission = df_new.Gearing_Type.unique()
gear=st.sidebar.selectbox("Transmission type (Gearing_Type)", transmission)
chaindrive = df_new.Drive_chain.unique()
chain = st.sidebar.selectbox("Chain drive (Drive_chain)", chaindrive)
cartype = df_new.Type.unique()
ctype = st.sidebar.selectbox("Car type", cartype)



button_styles = """
    <style>
        div.stButton > button {
            background-color: #3498db;  /* Set your desired background color */
            color: #ffffff;             /* Set your desired text color */
        }
    </style>
"""

# Display the button with custom styles
st.markdown(button_styles, unsafe_allow_html=True)

predict = st.button('Predict the car price')

filename = model
if model == 'Linear Regression':
    model = 'lasso_model'
else:
    model = 'DT_model'

model=pickle.load(open(model, "rb"))

my_dict = {"make_model":makemodel, 
           "hp_kW":hpower, 
           "km":km,
           "age":age, 
           "Gearing_Type":gear, 
           "Drive_chain":chain,
           "Type":ctype}

df_example = pd.DataFrame.from_dict([my_dict])

if predict:
    st.success(f'The estimated value of the car is {round(model.predict(df_example)[0])} Euro.')