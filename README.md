<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS116.N11.KHCL - LẬP TRÌNH PYTHON CHO MÁY HỌC -  PYTHON FOR MACHINE LEARNING</b></h1>

<a name="readme-top"></a>
## BẢNG MỤC LỤC
* [Giới thiệu môn học](#giới-thiệu-môn-học)
* [Giới thiệu nhóm](#giới-thiệu-nhóm)
* [Thư mục bài tập](#thư-mục-bài-tập)
* [Đồ án cuối kì](#đồ-án-cuối-kì)
<!--* [Tổng kết môn học](https://github.com/.../CS112.L21/blob/main/SummaryReport)-->


## GIỚI THIỆU MÔN HỌC
* **Tên môn học:** LẬP TRÌNH PYTHON CHO MÁY HỌC
* **Mã môn học:** CS116
* **Mã lớp:** CS116.N11.KHCL
* **Năm học:** HK1 (2022 - 2023)
* **Giảng viên:** TS. Nguyễn Vinh Tiệp

## GIỚI THIỆU NHÓM
| STT | Họ tên | MSSV | Vai trò | Email | Github | Facebook |
| :---: | --- | --- | --- | --- | --- | --- |
| 1 | Nguyễn Thị Như Vân | 20520855 | Thành viên | 20520855@gm.uit.edu.vn | [nhwzaan](https://github.com/nhwzaam) | [vanntn](https://www.facebook.com/xxnhwzaan/) |
| 2 | Huỳnh Anh Kiệt | 19521724 | Thành viên | 19521724@gm.uit.edu.vn |  |  |

## THƯ MỤC BÀI TẬP

_Lưu ý: bài tập làm cá nhân_

### [Bài tập 1: Các mô hình máy học cho bài toán Regression](https://github.com/nhwzaan/CS116/blob/main/Exercises/20520855_BT1.py)
Sử dụng các dataset đã được cung cấp để áp dụng các mô hình: Linear Regression, SVR với Polynomial kernel, Random Forest.
Tiến hành thí nghiệm và so sánh các phương pháp.

### [Bài tập 2: Xử lý dữ liệu GIS](https://github.com/nhwzaan/CS116/blob/main/Exercises/BT2_XuLiDuLieuGIS_20520855.ipynb)
  
  Bước 1: Cài đặt thư viện geopandas
  
  Bước 2: git clone https://github.com/CityScope/CSL_HCMC
  
  Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp
  
  Bước 4: hãy thực hiện các tác vụ truy vấn sau
  - Phường nào có diện tích lớn nhất
  - Phường nào có dân số 2019 (Pop_2019) cao nhất
  - Phường nào có diện tích nhỏ nhất
  - Phường nào có dân số thấp nhất (2019)
  - Phường nào có tốc độ tăng trưởng dân số nhanh nhất (dựa trên Pop_2009 và Pop_2019)
  - Phường nào có tốc độ tăng trưởng dân số thấp nhất
  - Phường nào có biến động dân số nhanh nhất
  - Phường nào có biến động dân số chậm nhất
  - Phường nào có mật độ dân số cao nhất (2019)
  - Phường nào có mật độ dân số thấp nhất (2019)


### [Bài tập 3: Trực quan hóa dữ liệu bản đồ](https://github.com/nhwzaan/CS116/blob/main/Exercises/20520855_BT3_TrucQuanHoaDuLieuBanDo.ipynb)

Bước 1: Cài đặt geopandas và folium

Bước 2: git clone https://github.com/CityScope/CSL_HCMC

Bước 3: dùng geopandas để đọc shapefile trong /Data/GIS/Population/population_HCMC/population_shapefile/Population_District_Level.shp

Bước 4: hãy thực hiện vẽ ranh giới các quận lên bản đồ dựa theo hướng dẫn sau:
https://geopandas.readthedocs.io/en/latest/gallery/polygon_plotting_with_folium.html


### [Bài tập 4: Gom cụm dữ liệu click của người dùng](https://github.com/nhwzaan/CS116/tree/main/Exercises/BT4_Click_Data_20520855_NguyenThiNhuVan)
Bước 1: Cài đặt các thư viện cần thiết

!pip install matplotlib==3.1.3

!pip install osmnet

!pip install folium

!pip install rtree

!pip install pygeos

!pip install geojson

!pip install geopandas

Bước 2: clone data từ https://github.com/CityScope/CSL_HCMC

Bước 3: Load ranh giới quận huyện và dân số quận huyện từ: Data\GIS\Population\population_HCMC\population_shapefile\Population_District_Level.shp

Bước 4: Load [dữ liệu click](https://github.com/nhwzaan/CS116/blob/main/Dataset/click_data.json) của người dùng 

Bước 5: Lọc ra 5 quận huyện có tốc độ tăng MẬT ĐỘ dân số nhanh nhất (Dùng dữ liệu 2019  và 2017)

Bước 6: Dùng spatial join (from geopandas.tools import sjoin) để lọc ra các điểm click của người dùng trong 5 quận/huyện hot nhất

Bước 7: chạy KMean cho top 5 quận huyện này. Lấy K = 20

Bước 8: Lưu 01 cụm điểm nhiều nhất trong các quận huyện ở Bước 5.

Bước 9: show lên bản đồ các cụm đông nhất theo từng quận huyện theo dạng HEATMAP

Bước 10: Lưu heatmap xuống file png

Lưu ý: Tải file notebook về và nộp lên Moodel

### [Bài tập 5: Linear Regression + đánh giá mô hình + Streamlit](https://github.com/nhwzaan/CS116/blob/main/Exercises/20520855_BT5_LinearRegressionWithStreamlit.py)

Các bạn nộp bài tập hồi quy tuyến tính sử dụng 2 phương pháp Train/Test split và K-fold cross validation để đánh giá mô hình.

Sử dụng Streamlit để làm giao diện ứng dụng theo gợi ý trên lớp lý thuyết.

Yêu cầu bao gồm:
Thiết kế giao diện với Streamlit để có thể:
  - Upload file csv (sau này có thể thay bằng tập dữ liệu khác dễ dàng).
  - Hiển thị bảng dữ liệu với file đã upload
  - Chọn lựa input feature (các cột dữ liệu đầu vào)
  - Chọn lựa hệ số cho train/test split: Ví dụ 0.8 có nghĩa là 80% để train và 20% để test
  - Chọn lựa hệ số K cho K-Fold cross validation: Ví dụ K =4
  - Nút "Run" để tiến hành chạy và đánh giá thuật toán

Output sẽ là biểu đồ cột hiển thị các kết quả sử dụng độ đo MAE và MSE. Lưu ý: Train/Test split và K-Fold cross validation được thực hiện độc lập, chỉ chọn 1 trong hai phương pháp này.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### [Bài tập 6: Phân lớp với Logistic Regression và đánh giá mô hình + Streamlit](https://github.com/nhwzaan/CS116/blob/main/Exercises/20520855_BT6_LogisticRegression_Streamlit.py)

Dataset: [Social Network Ads](https://github.com/nhwzaan/CS116/blob/main/Dataset/Social_Network_Ads.csv)

### [Bài tập 7: Classification với PCA để giảm số chiều](https://github.com/nhwzaan/CS116/blob/main/Exercises/20520855_BT7_PCA_Streamlit.py)

Sử dụng [Wine dataset](https://github.com/nhwzaan/CS116/blob/main/Dataset/wine.csv), kết hợp với streamlit:

- Bổ sung thêm option PCA, cho phép nhập số chiều sau khi giảm.

- Input feature X sau khi đã giảm chiều sẽ biến thành X'. Dùng X' để huấn luyện và dự đoán.

Lưu ý: Mô hình giảm số chiều được thực hiện trên tập train, thì sẽ giữ nguyên tham số để áp dụng trên tập test, chứ không fit lại trên tập test.

### [Bài tập 8: Phân loại văn bản với Naive Bayes](https://github.com/nhwzaan/CS116/blob/main/Exercises/BT8_TextClassificationNB.ipynb)

Dataset: [Restaurant Reviews](https://github.com/nhwzaan/CS116/blob/main/Dataset/Restaurant_Reviews.tsv)

### [Bài tập 9: CNN và các biến thể](https://github.com/nhwzaan/CS116/blob/main/Exercises/BT9_CNN_LeNet.ipynb)

Bước 1: Train và đánh giá trên tập test kiến trúc mạng LeNet với bộ MNIST

Bước 2: Thử thay đổi cấu hình của LeNet thành các biến thể. Một số gợi ý:

- Bỏ hết activation

- Bỏ hết Convolution layer

- Bỏ hết Pooling layer

Bước 3: đánh giá độ chính xác trên các biến thể trên

### [Bài tập 10: XGBoost](https://github.com/nhwzaan/CS116/blob/main/Exercises/20520855_BT10.py)

Sử dụng giao diện với Streamlit để so sánh các phương pháp phân loại với XGBoost.

XGBoost là thuật toán phân loại có hiệu quả cao được sử dụng trong các cuộc thi của Kaggle.

Chi tiết tham khảo thêm: https://xgboost.readthedocs.io/en/stable/

Hãy tiến hành cài đặt XGBoost Classifier và so sánh với một số phương ph như Logistic Regression, SVM, Decision Tree.

Dataset: [data](https://github.com/nhwzaan/CS116/blob/main/Dataset/Data.csv)

### [Bài tập 11: Chọn lựa mô hình với Grid Search](https://github.com/nhwzaan/CS116/blob/main/Exercises/20520855_BT11_GridSearchCV.py)

Với tham số mặc định của một mô hình, thông thường ta sẽ không đạt được độ chính xác cao nhất.

Để chọn lựa mô hình tốt nhất, ta sẽ tìm các siêu tham số hiệu quả nhất. Trong những phương pháp tìm tham số hiệu quả, Grid Search là một trong số đó.

Hãy sử dụng mô hình SVM (classification) trên tập dữ liệu Social Network Ads để tìm mô hình tốt  cho dữ liệu nêu trên sử dụng Grid Search.

Tham khảo code mẫu ở đây: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Dataset: [Social Network Ads](https://github.com/nhwzaan/CS116/blob/main/Dataset/Social_Network_Ads.csv)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ĐỒ ÁN CUỐI KÌ
**SENTIMENT ANALYSIS BASED ON MACHINE LEARNING METHODS**

- **_Abstract:_**  Sentiment analysis is an exciting new field of research in Artificial Intelligence combining Natural Language Processing, Machine Learning and Psychology. Since 2000, due to the proliferation of huge amounts of opinions in electronic form on the web, on social networks and on blogs, automatic means of polarity (positive, negative and neutral) detection in texts flourished in leaps and bounds. Individual and organizations with public interface can no longer afford to be oblivious of sentiments expressed about them in electronic form. In the present tutorial, we will first discuss the foundations of sentiment analysis, covering knowledge based and machine learning based techniques. Sentiment Analysis is the most common text classification tool that analyses an incoming message and tells whether the underlying sentiment is joy, sadness, anger, fear or neutral.
- _**Detailed report:**_ [Sentiment Analysis based on Machine Learning methods](https://github.com/nhwzaan/CS116/blob/main/Final%20Project/19521724_20520855.pdf)
- _**Dataset:**_ [training and testing data](https://github.com/nhwzaan/CS116/tree/main/Final%20Project/Dataset)
- **_Coding and supporting:_** please contact me at 20520855@gm.uit.edu.vn (≧∇≦)ﾉ

<p align="right">(<a href="#readme-top">back to top</a>)</p>

