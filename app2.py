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

tab1, tab4 = st.tabs(["", "UML"])
with tab4: 

            def read_objects():
                model_xgb = pickle.load(open('model_xgb.pkl','rb'))
                scaler = pickle.load(open('scaler.pkl','rb'))
                ohe = pickle.load(open('ohe.pkl','rb'))
                shap_values = pickle.load(open('shap_values.pkl','rb'))
                cats = list(itertools.chain(*ohe.categories_))
                return model_xgb, scaler, ohe, cats, shap_values

            model_xgb, scaler, ohe, cats, shap_values = read_objects()
        ##
            data = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
            with st.expander("UML"):
                st.title("Unsupervised Machine Learning")
                st.subheader('This will be a journey through the creation of UML customer segmentation, and an analysis of the obtained result.')
                'Let us start with the end result'
                st.image('https://raw.githubusercontent.com/Ceges98/BDS-Project/main/visualization.png', caption='not an optimal result')
                st.subheader('How did this come to be?')
                'To start the process of customer segmentation we need data regarding them.'
                data_raw = data.iloc[:, 0:7]
                st.write(data_raw.head(100))
                st.caption('these are the first 100 entrances in our relevant dataset, currently unfiltered.')
                'Some work is needed for this data to be operable in regards to UML, first we remove the unknown'
                data_raw = data_raw[data_raw["job"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["marital"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["education"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["housing"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["loan"].str.contains("unknown") == False]
                data_raw.drop('default', inplace=True, axis=1)
                data = data_raw
                tab01, tab02 = st.tabs(['new data', 'code'])
                with tab01:
                    st.write(data_raw.head(50))
                    st.caption('now there are no unknown values, we have also dropped the default column as it is almost solely "no" values and therefore should not be used to segment the customers.')
                with tab02:
                    drop_unknown = '''data_raw = data_raw[data_raw["job"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["marital"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["education"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["housing"].str.contains("unknown") == False]
                data_raw = data_raw[data_raw["loan"].str.contains("unknown") == False]
                data_raw.drop('default', inplace=True, axis=1)''' 
                    st.code(drop_unknown, language='python')
                'Next up is the fact that our data is unusable due to it being in a non-numerical format'
                'To fix this spread age out into 4 categories, replace yes/no with 1/0 on housing/loan, LabelEncode education and make a, admittedly subjective, list for jobs based on income'
                def age(data_raw):
                    data_raw.loc[data_raw['age'] <= 30, 'age'] = 1
                    data_raw.loc[(data_raw['age'] > 30) & (data_raw['age'] <= 45), 'age'] = 2
                    data_raw.loc[(data_raw['age'] > 45) & (data_raw['age'] <= 65), 'age'] = 3
                    data_raw.loc[(data_raw['age'] > 65) & (data_raw['age'] <= 98), 'age'] = 4 
                    return data_raw
                age(data_raw);
                data_raw = data_raw.replace(to_replace=['yes', 'no'], value=[1, 0])
                data_raw = data_raw.replace(to_replace=['unemployed', 'student', 'housemaid', 'blue-collar', 'services', 'retired', 'technician', 'admin.', 'self-employed', 'entrepreneur', 'management'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                labelencoder_X = LabelEncoder()  
                data_raw['education'] = data_raw['education'].replace({'illiterate':'a_illiterate'})
                data_raw['education'] = labelencoder_X.fit_transform(data_raw['education'])
                tab03, tab04 = st.tabs(['numeric data', 'code'])
                with tab03:
                    st.write(data_raw.head(50))
                
                with tab04:
                    numerification = '''def age(data_raw):
                    data_raw.loc[data_raw['age'] <= 30, 'age'] = 1
                    data_raw.loc[(data_raw['age'] > 30) & (data_raw['age'] <= 45), 'age'] = 2
                    data_raw.loc[(data_raw['age'] > 45) & (data_raw['age'] <= 65), 'age'] = 3
                    data_raw.loc[(data_raw['age'] > 65) & (data_raw['age'] <= 98), 'age'] = 4 
                    return data_raw
                age(data_raw);
                data_raw = data_raw.replace(to_replace=['yes', 'no'], value=[1, 0])
                data_raw = data_raw.replace(to_replace=['unemployed', 'student', 'housemaid', 'blue-collar', 'services', 'retired', 'technician', 'admin.', 'self-employed', 'entrepreneur', 'management'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                labelencoder_X = LabelEncoder()  
                data_raw['education'] = data_raw['education'].replace({'illiterate':'a_illiterate'})
                data_raw['education'] = labelencoder_X.fit_transform(data_raw['education'])'''
                    st.code(numerification, language='python')
                st.caption('this is not a perfect way of handling the issue but onehotencoding gave rise to different issues.')
                'It may be noted that marriage is currently untouched, this is due to troubles with OneHotEncoding. As such is was deemed unwise to throw in yet another subjective variable. It will therefor be dropped.'
                data_raw = data_raw.drop(columns = 'marital')
                st.write(data_raw.head())
                'lastly these numbers need to be scaled'
                data_raw_scaled = scaler.fit_transform(data_raw)
                tab05, tab06 = st.tabs(['scaled data', 'code'])
                with tab05:
                    st.write(data_raw_scaled[:10])
                with tab06:
                    scaled_date = '''data_raw_scaled = scaler.fit_transform(data_raw)'''
                    st.code(scaled_date, language='python')
                st.caption('Now the previous sizes of the values have been standard scaled.')
                'From here on out the process will be shown through code with comments'
                rest = '''#umap accepts standard-scaled data
    embeddings = umap_scaler.fit_transform(data_raw_scaled)
    #we choose 6 clusters
    clusterer = KMeans(n_clusters=6)
    Sum_of_squared_distances = []
    K = range(1,10)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_raw_scaled)
        Sum_of_squared_distances.append(km.inertia_)
    #no clear elbow
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    #we fit clusters on our scaled data
    clusterer.fit(data_raw_scaled)
    #we then copy the clusters into the original file
    data1['cluster'] = clusterer.labels_
    #can use the clusters to fx. see the mean of age in our clusters.
    #note that age does not seem a big factor in clustering as the mean is mostly the same.
    data1.groupby('cluster').age.mean()
    #prepping our vis_data
    vis_data = pd.DataFrame(embeddings)
    vis_data['cluster'] = data1['cluster']
    vis_data['education'] = data1['education']
    vis_data['age'] = data1['age']
    vis_data['job'] = data1['job']
    vis_data['marital'] = data1['marital']
    vis_data['housing'] = data1['housing']
    vis_data['loan'] = data1['loan']
    vis_data.columns = ['x', 'y', 'cluster','education', 'age', 'job', 'marital', 'housing', 'loan']
    #finally plotting the data with relevant tooltips
    #for unknown reasons a null cluster is made alongside our other clusters
    alt.data_transformers.enable('default', max_rows=None)
    alt.Chart(vis_data).mark_circle(size=60).encode(
        x='x',
        y='y',
        tooltip=['education', 'age', 'job', 'marital', 'housing', 'loan'],
        color=alt.Color('cluster:N', scale=alt.Scale(scheme='dark2')) #use N after the var to tell altair that it's categorical
    ).interactive()'''
                st.code(rest, language='python')
                'The reasoning behind showing this block of code is mainly to show the procedure that was taken following the data-preprocessing and showing a more in-depth process is not very useful as the end result is flawed.'
                'Speaking of, here we have once again the result so that the flaws can be discussed'
                st.image('https://raw.githubusercontent.com/Ceges98/BDS-Project/main/visualization.png', caption='still not optimal')
                '''To understand the flaws we have to look at the goal of the model. 
                The goal of this model was to place the customers in clusters based on their data.
                As such there are two problems:
                1. The clusters are randomly dispersed.
                2. An extra null-cluster has been created.
                Optimally we would be able to find and fix the problem causing these flaws but as of know this model
                has presented a learning opportunity and not a finished piece of work.'''