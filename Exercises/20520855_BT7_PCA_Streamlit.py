import streamlit as st
import pandas as pd
import numpy as np
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn import datasets

st.title("Classification With PCA Application")

#st.markdown("# Data Upload")
data = pd.DataFrame()

st.markdown("# Methods ♪(´▽｀)")

lgr = LogisticRegression()


def data_select(df):
    Data_flexible_column = {}
    
    for col_name in cols[:-1]:
        col_checkbox = st.checkbox(col_name)
        if col_checkbox:
            Data_flexible_column[col_name] = df[col_name]

    Data_flexible_column[cols[-1]] = df[cols[-1]]
    Data_use_for_model = pd.DataFrame(Data_flexible_column)    
    
    st.write("Data Selected")
    st.write(Data_use_for_model)
    return Data_use_for_model



def Logistic_PCA(df, n_components):
    

    df = data_select(df)
    
    X_pca = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    
    pca = PCA(n_components)
    pca.fit(X_pca)
    X_pca = pca.transform(X_pca)
    
    train_select = st.slider("Choose percentage of train size",0.0,1.0,0.8,0.1)
    X_train,X_test,y_train,y_test = train_test_split(X_pca,y,train_size=train_select,random_state=0)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    accuracy = model.score(X_test, y_test)
    st.write("Accuracy ", accuracy.round(2))
    st.write("Precision: ", precision_score(y_test, y_predict, average='macro'))
    st.write("Recall: ", recall_score(y_test, y_predict, average='macro'))
    st.write("F1-score: ", f1_score(y_test, y_predict, average='macro'))
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plot_confusion_matrix(model, X_test, y_test)
    st.pyplot()


def Logistic(df):
    st.markdown("# Choose input feature ♪(´▽｀)")
    cols_1 = st.multiselect('Input feature you want:', data.columns, default=[])
    st.write('You selected:', cols_1)

    name_cols = []
    for i in cols_1:
        name_cols.append(i)
    multi_cols = data[name_cols]

    st.markdown("# Ouput feature ♪(´▽｀)")
    st.subheader("Output feature mặc định là cột cuối cùng của file csv ^^")
    last_column = data.iloc[: , -1:]
    st.dataframe(last_column, width=700, height=300)


    lgr = LogisticRegression()


    st.subheader("Split data ♪(´▽｀)")
    ratio = st.number_input('Choose the ratio between trainning data and testing data')
    st.write('The data will be split into ', ratio*100, "% for training and the rest for testing")
    if st.button('RUN'):
        if ratio != 0:
            ts = float(1-ratio)
            X_train, X_test, y_train, y_test = train_test_split(multi_cols, last_column, test_size=ts, random_state=42)
            lgr.fit(X_train, y_train)
            y_pred = lgr.predict(X_test)

            #confusion matrix
            accuracy = lgr.score(X_test, y_test)
            st.write("Accuracy ", accuracy.round(4))
            st.write("Precision: ", precision_score(y_test, y_pred, average='macro'))
            st.write("Recall: ", recall_score(y_test, y_pred, average='macro'))
            st.write("F1-score: ", f1_score(y_test, y_pred, average='macro'))
        
        
            
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plot_confusion_matrix(lgr, X_test, y_test)
        st.pyplot()
        #plt.show()
        


check1 = st.checkbox("Logistic")
check2 = st.checkbox("Logistic PCA")


if check1:
    uploaded_file = st.file_uploader("Choose a file ♪(´▽｀)")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open("data/" + uploaded_file.name, "wb") as f:
            f.write(bytes_data)
        
        data = pd.read_csv('data/' + uploaded_file.name, sep=',')

    Logistic(data)
        
        
if check2:
    wine = datasets.load_wine()
    df = pd.DataFrame(data= np.c_[wine['data'], wine['target']],
                     columns= wine['feature_names'] + ['target'])
    cols=df.columns
    st.write(df)
    n_components = st.number_input('Number component: ')
    n_components = int(n_components)
    Logistic_PCA(df, n_components)
    