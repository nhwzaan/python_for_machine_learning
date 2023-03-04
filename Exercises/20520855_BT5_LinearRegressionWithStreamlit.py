#Thông tin sinh viên:
# - Họ và tên: Nguyễn Thị Như Vân
# - MSSV: 20520855
# - Lớp: CS116.N11.KHCL

from audioop import mul
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.axes as mltaxes

st.title("Regresion Application")

st.markdown("# Data Upload")

#B1: Upload file
uploaded_file = st.file_uploader("Choose a .csv file ♪(´▽｀)")

data = pd.DataFrame()

#B2: Hiển thị table lên giao diện 
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open("data/" + uploaded_file.name, "wb") as f:
        f.write(bytes_data)

    data = pd.read_csv("data/" + uploaded_file.name)
    st.dataframe(data, width=700, height=300)


#B3: Chọn (ra cột để làm) input feature (tạo option checkbox A B C D...)
st.markdown("# Choose input feature ♪(´▽｀)")
cols_1 = st.multiselect('Input feature you want:', data.columns, default=[])
st.write('You selected:', cols_1)

name_cols = []
for i in cols_1:
    name_cols.append(i)
multi_cols = data[name_cols]

    
#B4: Output feature mặc định là cột cuối cùng của file csv
st.markdown("# Ouput feature ♪(´▽｀)")
st.subheader("Output feature mặc định là cột cuối cùng của file csv ^^")
last_column = data.iloc[: , -1:]
st.dataframe(last_column, width=700, height=300)


#B5: Chọn phương pháp thực hiện: train test split hoặc k-fold
st.markdown("# Choose methods")
mtds = st.radio('Chào mừng bạn đến với chương trình "Hãy chọn giá đúng" ', ('Train test split', 'K-fold Cross Validation'))

lr = LinearRegression()

if mtds == 'Train test split':
    st.subheader("Split data ♪(´▽｀)")
    #ratio = 0.8
    ratio = st.number_input('Choose the ratio between trainning data and testing data')
    st.write('The data will be split into ', ratio*100, "% for training and the rest for testing")
    if st.button('RUN'):
        if ratio != 0:
            ts = float(1-ratio)
            X_train, X_test, y_train, y_test = train_test_split(multi_cols, last_column, test_size=ts, random_state=42)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            st.write("MAE:", mae)
            st.write("MSE:", mse)

            #visualize
            plt.figure()
            plt.bar(1 - 0.2, mae, 0.4, label = 'MAE')
            plt.bar(1 + 0.2, mse, 0.4, label = 'MSE')
            plt.yscale('log')
            plt.xlabel("Cost")
            plt.ylabel("Value")
            plt.title("MAE and MSE")
            plt.legend()
            fig = plt.gcf()
            st.pyplot(fig)
        #st.write(y_pred)


else:
    st.subheader("k-fold cross validation ♪(´▽｀)")
    k = st.slider('Choose the number of k', 1, 10, 4)
    st.write("This model will be checked with ", k, "-fold cross validation ^^!")
    kf_mse, kf_mae = [], []
    
    if st.button('RUN'):
        kfold = KFold(n_splits=int(k), shuffle=True)
        #index = 1
        
        for train_index, test_index in kfold.split(multi_cols, last_column):
            
            lr.fit(multi_cols[train_index[0]:train_index[-1]], last_column[train_index[0]:train_index[-1]])

            y_predict_with_kfold = lr.predict(multi_cols[test_index[0]:test_index[-1]])
            
            kf_mae.append(mean_squared_error(last_column[test_index[0]:test_index[-1]], y_predict_with_kfold))
            kf_mse.append(mean_absolute_error(last_column[test_index[0]:test_index[-1]], y_predict_with_kfold))
    st.write("MSE:", kf_mse)    
    st.write("MAE:", kf_mae)

    #visualize
    X = range(len(kf_mae))
    X_axis = np.arange(len(X))

    plt.figure()
    plt.bar(X_axis - 0.2, kf_mse, 0.4, label = 'MSE')
    plt.bar(X_axis + 0.2, kf_mae, 0.4, label = 'MAE')
    plt.yscale('log')
    plt.xticks(X_axis, X)
    plt.xlabel("Cost")
    plt.ylabel("Value")
    plt.title("MAE and MSE")
    plt.legend()
    fig = plt.gcf()
    st.pyplot(fig)








