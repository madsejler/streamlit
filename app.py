# Load packages (comments for more special stuff)

import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model...and therefore need to load it
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import shap # add prediction explainability

import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st
from streamlit_shap import st_shap # wrapper to display nice shap viz in the app
import plotly.express as px
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

##Streamlit interface:
st.set_page_config(page_title='Market Values - Prediction',
                    page_icon="💰",
                    layout='wide')

colT1,colT2 = st.columns([10,20])
with colT2:
   st.title('Prediction of real market value 💰')

data = pd.read_csv("https://raw.githubusercontent.com/madsejler/streamlit/main/dataEDA.csv", sep=";")
differences = pd.read_csv("https://raw.githubusercontent.com/madsejler/streamlit/main/differences.csv", sep=",")

tab1, tab2, tab3 = st.tabs(["Data Exploration","Real Price Information", "SML Model Comparison"])
with tab1:

    # dashboard title

    st.title("Data Dashboard")

    job_filter = st.selectbox("Select the Job", pd.unique(data['Competition']))


    # creating a single-element container.
    placeholder = st.empty()
    # dataframe filter 
    data = data[data['Competition']==job_filter]

    # near real-time / live feed simulation 
    for seconds in range(10):
        with placeholder.container(): 

            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                st.markdown("Age/MarketValue heatmap")
                fig = px.density_heatmap(data_frame=data, y = data['Age'], x = data['market_value'])
                st.write(fig)
            with fig_col2:
                st.markdown("Age distribution")
                fig2 = px.histogram(data_frame = data, x = data['Age'])
                st.write(fig2)
            st.markdown("### Detailed Data View")
            st.dataframe(data.iloc[:,:61])
            time.sleep(1)
    with tab2:

        st.title('Real price information')
        st.markdown("Real Value vs. Market Value")
        player_filter = st.selectbox("Select the Player", pd.unique(data['Player Name']))
        data = data[data['Player Name']==player_filter]
        st.dataframe(data.filter(items=['Player Name','Squad', 'Prediction', 'market_value', 'difference', 'difference %']))        
        time.sleep(1)
        
        st.markdown("")
        st.markdown("Statistics")
        data = data[data['Player Name']==player_filter]
        st.dataframe(data.iloc[:,:61])        
        time.sleep(1)
        
        
        with tab3:
            st.subheader("SML Model Accuracy")
            st.markdown("On this tab, we will explain why we used the XGB-model, and what parameters we made the decision on")
            with st.expander("What is the method for comparing the choosing of the three models?"):
                st.markdown(""" The method for comparing, is done through running the notebook five times 
                and chekcing how much the accuracy of the three models each time. We end up with five values
                for each model, and then we are comparing the mean, to which model overall performs the best.
                There are several random elements in the code, for instance we undersample the data in order
                to balance it for SML purposes. The undersample is done randomly and everytime we run the code
                the undersample will include different obersvations. Further the train-test split consist of a 
                random element. By running the note 5 times we expect the deviation to be under of the accuracy
                to be under 5%. Take a look at this page to see the documentation for the values of the 5 runs.
                """)
            st.subheader("Logistic Regression")
            st.markdown("The five values for this model had a range of 0,78% which a good deal under the 5% 🤓 ")

            st.subheader("XGBClassifier")
            st.markdown("The five values for this model had a range of 2,23% which is also under 5% 🤗 ")

            st.subheader("RandomForrester")
            st.markdown("The five values for this model had a range of 1,23% which is also under 5% 🎉 ")

            st.subheader("Ranking of the models by the mean accuracy of the 5 runs")
            st.markdown("1. **Logistic Regression**: 74,50% 🏆" )
            st.markdown("2. **XGB Classifier**: 73,44% 🥈" )
            st.markdown("3. **Random Forest**: 71,30% 🥉" )

            st.markdown("""Due to some technical issues with the Logistic regression, we decided to use the XGB Classifier
            for the model anyways, because the LR-model seems to do limited ietrations on the training data. We did not 
            have that problem with the XGB-model, so we went ahead and used the XGB for the prediction model on this webpage """)