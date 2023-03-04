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


st.title("Classification Application")

st.markdown("# Data Upload")
data = pd.DataFrame()
columns = defaultdict(list)

uploaded_file = st.file_uploader("Choose a .csv file ♪(´▽｀)")


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open("data/" + uploaded_file.name, "wb") as f:
        f.write(bytes_data)
    
    data = pd.read_csv('data/' + uploaded_file.name, sep=';')
    st.dataframe(data, width=700, height=300)

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
        st.write("Precision: ", precision_score(y_test, y_pred))
        st.write("Recall: ", recall_score(y_test, y_pred))
        st.write("F1-score: ", f1_score(y_test, y_pred))
        st.write("Log loss: ", log_loss(y_test, y_pred))
       
       
        
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plot_confusion_matrix(lgr, X_test, y_test)
    st.pyplot()
     #plt.show()