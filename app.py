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

tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration","Real Price Information", "Predict", "SML Model Comparison"])
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
            st.dataframe(data.iloc[:,1:59])
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
        st.dataframe(data.iloc[:,1:59])        
        time.sleep(1)
        
        with tab3:
            st.subheader("Predict")
            st.markdown("")

            st.title('Will this given costumer say yes?')

            model_xgb = pickle.load(open('model_xgb.pkl','rb')

            # @st.experimental_singleton
            # def read_objects():
            #     model_xgb = pickle.load(open('model_xgb.pkl','rb'))
            #     scaler = pickle.load(open('scaler.pkl','rb'))
            #     ohe = pickle.load(open('ohe.pkl','rb'))
            #     shap_values = pickle.load(open('shap_values.pkl','rb'))
            #     cats = list(itertools.chain(*ohe.categories_))
            #     return model_xgb, scaler, ohe, cats, shap_values

            # model_xgb, scaler, ohe, cats, shap_values = read_objects()


            with st.expander("What's the purpose of this app?"):
                st.markdown("""
                This app will help you determine if you should call a given costumer! 💵 💴 💶 💷
                It can further help you reconsider your strategic approach to the costumer,
                in the case that our SML model will predict a "No" from the costumer.
                """)

            st.title('Costumer description')

            #Below all the bank client's info will be selected
            st.subheader("Select the Customer's Age")
            age = st.slider("", min_value = 17, max_value = 98, 
                                    step = 1, value = 41)
            st.write("Selected Age:", age)

            st.subheader("Select the Customer's Jobtype")
            job = st.radio("", ohe.categories_[0])
            st.write("Selected Job:", job)

            st.subheader("Select the Customer's Marital")
            marital = st.radio("", ohe.categories_[1])
            st.write("Selected Marital:", marital)

            st.subheader("Select the Customer's Education")
            education = st.radio("", data['education'].unique())
            st.write("Selected Education:", education)
            #Defining a encoding function for education
            def encode_education(selected_item):
                dict_education = {'basic.4y':1, 'high.school':4, 'basic.6y':2, 'basic.9y':3, 'professional.course':5, 'university.degree':6, 
            'illiterate':0}
                return dict_education.get(selected_item)
            ### Using function for encoding on education
            education = encode_education(education) 

            poutcome = st.selectbox('What was the previous outcome for this costumer?', options=ohe.categories_[4])
            campaign = st.number_input('How many contacts have you made for this costumer for this campagin already?', min_value=0, max_value=35)
            previous = st.number_input('How many times have you contacted this client before?', min_value=0, max_value=35)

            #Button for predicting the costumers answer
            if st.button('Deposit Prediction 💵'):

                # make a DF for categories and transform with one-hot-encoder
                new_df_cat = pd.DataFrame({'job':job,
                            'marital':marital,
                            'month': 'oct', #This could be coded with a date.today().month function
                            'day_of_week':'fri', #This could aswell be coded with a function
                            'poutcome':poutcome}, index=[0])
                new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = cats , index=[0])

                # make a DF for the numericals and standard scale
                new_df_num = pd.DataFrame({'age':age, 
                                        'education': education,
                                        'campaign': campaign,
                                        'previous': previous, 
                                        'emp.var.rate': 1.1, #This could be scraped from a site like Statistics Portugal
                                        'cons.price.idx': 93.994, #This could be scraped from a site like Statistics Portugal
                                        'cons.conf.idx': -36.4, #This could be scraped from a site like Statistics Portugal
                                        'euribor3m': 4.857, #This could be scraped from a site like Statistics Portugal
                                        'nr.employed': 5191.0 #This could be scraped from a site like Statistics Portugal
                                    }, index=[0])
                new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
                
                #Bringing all columns together
                line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

                #Run prediction for the new observation. Inputs to this given above
                predicted_value = model_xgb.predict(line_to_pred)[0]
                
                
                #Printing the result
                st.metric(label="Predicted answer", value=f'{predicted_value}')
                st.subheader(f'What does {predicted_value} mean? 1 equals to yes, while 0 equals to no')

                #Printing SHAP explainer
                st.subheader(f'Lets explain why the model predicts the output above! See below for SHAP value:')
                shap_value = explainer.shap_values(line_to_pred)
                st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=900)
        
                with tab4:
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
                
