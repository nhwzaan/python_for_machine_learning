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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


st.title("SVM Hyperparameter Tuning using GridSearchCV")

st.markdown("# Data Upload")
df = pd.DataFrame()
columns = defaultdict(list)

uploaded_file = st.file_uploader("Choose a .csv file ♪(´▽｀)")


if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open("data/" + uploaded_file.name, "wb") as f:
        f.write(bytes_data)
    
    df = pd.read_csv('data/' + uploaded_file.name, sep=';')
    st.dataframe(df, width=700, height=300)

st.subheader("# Data keys")
st.write(df.keys())

st.markdown("# Choose input feature ♪(´▽｀)")
cols_1 = st.multiselect('Input feature you want:', df.columns, default=[])
st.write('You selected:', cols_1)

name_cols = []
for i in cols_1:
    name_cols.append(i)
multi_cols = df[name_cols]

st.markdown("# Ouput feature ♪(´▽｀)")
st.subheader("Output feature mặc định là cột cuối cùng của file csv ^^")
last_column = df.iloc[: , -1:]
st.dataframe(last_column, width=700, height=300)

# Models
svc = SVC()


st.subheader("Split data ♪(´▽｀)")
ratio = st.number_input('Choose the ratio between trainning data and testing data')
st.write('The data will be split into ', ratio*100, "% for training and the rest for testing")
if st.button('RUN'):
    if ratio != 0:
        ts = float(1-ratio)
        X_train, X_test, y_train, y_test = train_test_split(multi_cols, last_column, test_size=ts, random_state=42)
        

        st.subheader("SVM Classifier: ")
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        accuracy = svc.score(X_test, y_test)
        st.write("Accuracy ", accuracy.round(5))
        st.write("Precision: ", precision_score(y_test, y_pred, average='macro'))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro'))
        st.write("F1-score: ", f1_score(y_test, y_pred, average='macro'))
            
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plot_confusion_matrix(svc, X_test, y_test)
        st.pyplot()


        st.subheader("Using GridSearchCV to adjust hyperparameter")
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']}

        grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3)
        grid.fit(X_train, y_train)


        st.write("The best parameter after tuning: ",grid.best_params_)

        st.write("SVC model after hyperparameter tunning: ", grid.best_estimator_)

        grid_predictions = grid.predict(X_test)

        # print classification report
        st.code(classification_report(y_test, grid_predictions))


