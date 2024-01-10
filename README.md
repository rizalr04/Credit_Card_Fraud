# Credit_Card_Fraud
Credit card fraud adalah tindakan atau usaha untuk mendapatkan keuntungan secara tidak sah dengan menggunakan informasi kartu kredit orang lain tanpa izin. Ini dapat melibatkan aktivitas seperti pembelian barang atau layanan tanpa pengetahuan atau persetujuan pemilik kartu, atau bahkan mencuri informasi kartu kredit untuk melakukan transaksi ilegal. 

Dalam proyek ini, saya akan membangun model machine learning yang dapat mengidentifikasi apakah suatu transaksi dengan kartu kredit bersifat penipuan atau tidak menggunakan beberapa algoritma dan teknik klasifikasi. Tujuan utama proyek ini adalah menciptakan model yang dapat membantu mengamankan transaksi keuangan dengan mengenali pola-pola yang terkait dengan aktivitas penipuan pada kartu kredit.

## Project Outline
1. Dataset
2. Data Preprocessing
3. Model Building
4. Conclusion

## 1. Dataset
Dataset credit card fraud ini merupakan data yang diambil dari website kaggle.com.
Link Dataset [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset ini berisi transaksi yang dilakukan dengan kartu kredit oleh pemegang kartu Eropa pada bulan September 2013. Dataset ini mencakup transaksi yang terjadi dalam dua hari, di mana terdapat 492 kasus penipuan dari total 284,807 transaksi. Dataset ini sangat tidak seimbang, dengan kelas positif (penipuan) menyumbang 0.172% dari seluruh transaksi.

Dataset ini hanya berisi variabel input numerik yang merupakan hasil dari transformasi PCA (Principal Component Analysis). Sayangnya, karena masalah kerahasiaan, kami tidak dapat menyediakan fitur-fitur asli dan informasi latar belakang lebih lanjut tentang data ini. Fitur V1, V2, ..., V28 adalah komponen utama yang diperoleh dengan PCA, satu-satunya fitur yang tidak mengalami transformasi PCA adalah 'Time' dan 'Amount'. Fitur 'Time' berisi detik yang berlalu antara setiap transaksi dan transaksi pertama dalam dataset. Fitur 'Amount' adalah Jumlah transaksi, fitur ini dapat digunakan, misalnya, untuk pembelajaran yang sensitif terhadap biaya yang tergantung pada kasus. Fitur 'Class' adalah variabel respons dan bernilai 1 dalam kasus penipuan dan 0 untuk transaksi lainnya.

### Loading Dataset
```python
import numpy as np
import pandas as pd
```
```python
from google.colab import drive
drive.mount("/content/drive/")
```
```python
df = pd.read_csv('/content/drive/MyDrive/Data Portofolio/creditcard.csv')
pd.options.display.max_columns = None
df.head()
```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/data.PNG)

## 2. Data Preprocessing
- Mengecek informasi dalam data
```python
df.info()
```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/df%20info.PNG)
- Mengecek nilai kosong pada data
```python
df.isnull().sum()
```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/df%20isnull.PNG)
- Mengubah skala pada kolom ‘Amount’ agar memiliki nilai yang seragam dengan kolom lainnya menggunakan standard scaler
```python
sc = StandardScaler()
df['Amount']=sc.fit_transform(pd.DataFrame(df['Amount']))
```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/standardscaller.PNG)
- Menghapus kolom 'Time'
```python
df=df.drop(['Time'], axis=1)
```
- Mengecek keseimbangan data pada kolom 'Class' dimana kolom 'Class' merupakan kolom target untuk training model
```python
import seaborn as sns

sns.countplot(x ='Class', data = df)
```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/imbalanced%20data.png)

## 3. Model Building
```python
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

### Handling Imbalanced Dataset
**1. Undersampling & Building Model**

   **Undersampling**
   ```python
   normal = df[df['Class']==0]
   fraud = df[df['Class']==1]
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/us%20shape.PNG)
   ```python
   normal_sample = normal.sample(n=473)
   df_undersampling = pd.concat([normal_sample,fraud], ignore_index=True)
   sns.countplot(x ='Class', data = df_undersampling)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/class%20us.png)
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/class%20us%20shape.PNG)

   **Model Building**
   ```python
   X = df_undersampling.drop('Class',axis=1)
   y = df_undersampling['Class']
   ```
   ```python
   from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                 random_state=42)
   ```
   **Logistic Regression**
   ```python
   log_us = LogisticRegression()
   log_us.fit(X_train,y_train)
   ```
   ```python
   y_pred1_us = log_us.predict(X_test)

   accuracy_log_us = accuracy_score(y_test, y_pred1_us)
   print("Accuracy   :", accuracy_log_us)
   precision_log_us = precision_score(y_test, y_pred1_us)
   print("Precision :", precision_log_us)
   recall_log_us = recall_score(y_test, y_pred1_us)
   print("Recall    :", recall_log_us)
   f1_score_log_us = f1_score(y_test, y_pred1_us)
   print("F1_score  :", f1_score_log_us)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/lr%20us.PNG)
   
   **Random Forest Classifier**
   ```python
   rf_us= RandomForestClassifier()
   rf_us.fit(X_train,y_train)
   ```
   ```python
   y_pred2_us = rf_us.predict(X_test)

   accuracy_rf_us = accuracy_score(y_test, y_pred2_us)
   print("Accuracy   :", accuracy_rf_us)
   precision_rf_us = precision_score(y_test, y_pred2_us)
   print("Precision :", precision_rf_us)
   recall_rf_us = recall_score(y_test, y_pred2_us)
   print("Recall    :", recall_rf_us)
   f1_score_rf_us = f1_score(y_test, y_pred2_us)
   print("F1_score  :", f1_score_rf_us)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/rf%20us.PNG)
   
   **Support Vector Machine**
   ```python
   svm_us = SVC()
   svm_us.fit(X_train,y_train)
   ```
   ```python
   y_pred3_us = svm_us.predict(X_test)

   accuracy_svm_us = accuracy_score(y_test, y_pred3_us)
   print("Accuracy   :", accuracy_svm_us)
   precision_svm_us = precision_score(y_test, y_pred3_us)
   print("Precision :", precision_svm_us)
   recall_svm_us = recall_score(y_test, y_pred3_us)
   print("Recall    :", recall_svm_us)
   f1_score_svm_us = f1_score(y_test, y_pred3_us)
   print("F1_score  :", f1_score_svm_us)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/svm%20us.PNG)
   
   **XGBoost**
   ```python
   xgb_us = XGBClassifier()
   xgb_us.fit(X_train,y_train)
   ```
   ```python
   y_pred4_us = xgb_us.predict(X_test)

   accuracy_xgb_us = accuracy_score(y_test, y_pred4_us)
   print("Accuracy   :", accuracy_xgb_us)
   precision_xgb_us = precision_score(y_test, y_pred4_us)
   print("Precision :", precision_xgb_us)
   recall_xgb_us = recall_score(y_test, y_pred4_us)
   print("Recall    :", recall_xgb_us)
   f1_score_xgb_us = f1_score(y_test, y_pred4_us)
   print("F1_score  :", f1_score_xgb_us)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/xgb%20us.PNG)
   ```python
   from prettytable import PrettyTable
   print("\t\t\t\tUndersampling")
   Result_table = PrettyTable(["S.No.","Model","Accuracy","Precison","Recall","F1 Score"])
   Result_table.add_row(result1_us)
   Result_table.add_row(result2_us)
   Result_table.add_row(result3_us)
   Result_table.add_row(result4_us)
   print(Result_table)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/result%20us.PNG)
   ```python
   sns.barplot(x ='Models', y='Acc', data = final_us)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/acc%20us.png)

**2. Oversampling & Building Model**

   **Oversampling**
   ```python
   X = df.drop('Class',axis=1)
   y = df['Class']
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/os%20shape.PNG)
   ```python
   from imblearn.over_sampling import SMOTE

   X_res,y_res = SMOTE().fit_resample(X,y)
   sns.countplot(x =y_res)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/class%20os.png)
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/class%20os%20shape.PNG)

   **Model Building**
   ```python
   from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,
                                                 random_state=42)
   ```
   **Logistic Regression**
   ```python
   log_os = LogisticRegression()
   log_os.fit(X_train,y_train)
   ```
   ```python
   y_pred1_os = log_os.predict(X_test)

   accuracy_log_os = accuracy_score(y_test, y_pred1_os)
   print("Accuracy   :", accuracy_log_os)
   precision_log_os = precision_score(y_test, y_pred1_os)
   print("Precision :", precision_log_os)
   recall_log_os = recall_score(y_test, y_pred1_os)
   print("Recall    :", recall_log_os)
   f1_score_log_os = f1_score(y_test, y_pred1_os)
   print("F1_score  :", f1_score_log_os)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/lr%20os.PNG)
   
   **Random Forest Classifier**
   ```python
   rf_os= RandomForestClassifier()
   rf_os.fit(X_train,y_train)
   ```
   ```python
   y_pred2_os = rf_os.predict(X_test)

   accuracy_rf_os = accuracy_score(y_test, y_pred2_os)
   print("Accuracy   :", accuracy_rf_os)
   precision_rf_os = precision_score(y_test, y_pred2_os)
   print("Precision :", precision_rf_os)
   recall_rf_os = recall_score(y_test, y_pred2_os)
   print("Recall    :", recall_rf_os)
   f1_score_rf_os = f1_score(y_test, y_pred2_os)
   print("F1_score  :", f1_score_rf_os)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/rf%20os.PNG)
   
   **Support Vector Machine**
   ```python
   svm_os = SVC()
   svm_os.fit(X_train,y_train)
   ```
   ```python
   y_pred3_os = svm_os.predict(X_test)

   accuracy_svm_os = accuracy_score(y_test, y_pred3_os)
   print("Accuracy   :", accuracy_svm_os)
   precision_svm_os = precision_score(y_test, y_pred3_os)
   print("Precision :", precision_svm_os)
   recall_svm_os = recall_score(y_test, y_pred3_os)
   print("Recall    :", recall_svm_os)
   f1_score_svm_os = f1_score(y_test, y_pred3_os)
   print("F1_score  :", f1_score_svm_os)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/svm%20os.PNG)
   
   **XGBoost**
   ```python
   xgb_os = XGBClassifier()
   xgb_os.fit(X_train,y_train)
   ```
   ```python
   y_pred4_os = xgb_os.predict(X_test)

   accuracy_xgb_os = accuracy_score(y_test, y_pred4_os)
   print("Accuracy   :", accuracy_xgb_os)
   precision_xgb_os = precision_score(y_test, y_pred4_os)
   print("Precision :", precision_xgb_os)
   recall_xgb_os = recall_score(y_test, y_pred4_os)
   print("Recall    :", recall_xgb_os)
   f1_score_xgb_os = f1_score(y_test, y_pred4_os)
   print("F1_score  :", f1_score_xgb_os)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/xgb%20os.PNG)
   ```python
   from prettytable import PrettyTable
   print("\t\t\t\tOversampling")
   Result_table = PrettyTable(["S.No.","Model","Accuracy","Precison","Recall","F1 Score"])
   Result_table.add_row(result1_os)
   Result_table.add_row(result2_os)
   Result_table.add_row(result3_os)
   Result_table.add_row(result4_os)
   print(Result_table)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/result%20os.PNG)
   ```python
   sns.barplot(x ='Models', y='Acc', data = final_os)
   ```
   ![image](https://github.com/rizalr04/Credit_Card_Fraud/blob/280e8c9cbe0007df812ddab5de4c66e0b609017f/Asset/acc%20os.png)

   ## 4. Conclusion
   1. Dengan menggunakan teknik undersampling dalam penanganan ketidakseimbangan data dalam proyek ini dapat dilihat bahwa Logistic Regression merupakan model terbaik berdasarkan metrik untuk mengklasifikasikan model. Logistic Regression memiliki akurasi 94% yang menunjukan bahwa model ini dapat mengidentifikasi 94% transaksi apakah bersifat penipuan atau tidak dengan nilai recall sebesar 91%, menunjukkan jumlah True Positives yang tinggi, yang didukung oleh F1-score yang tinggi sebesar 94%.
   2. Dengan menggunakan teknik oversampling metode SMOTE dalam penanganan ketidakseimbangan data dalam proyek ini dapat dilihat bahwa Random Forest Classifier dan XGBoost merupakan model terbaik berdasarkan metrik dengan nilai yang sama tinggi untuk mengklasifikasikan model. Random Forest Classifier dan XGBoost memiliki akurasi 99% yang menunjukan bahwa model ini dapat mengidentifikasi 99% transaksi apakah bersifat penipuan atau tidak dengan nilai recall sebesar 100%, menunjukkan jumlah True Positives yang tinggi, yang didukung oleh F1-score yang tinggi sebesar 99%.
   3. Dalam proyek ini, penanganan ketidakseimbangan data menggunakan teknik oversampling dengan metode SMOTE telah terbukti memberikan hasil yang lebih baik daripada undersampling. Hal ini dapat dilihat dari peningkatan kinerja model machine learning dalam mendeteksi transaksi penipuan, di mana oversampling memberikan jumlah sampel yang lebih banyak untuk kelas minoritas, meningkatkan kemampuan model untuk memahami pola dan mendeteksi kasus penipuan dengan lebih akurat. Sebaliknya, undersampling dapat menghilangkan informasi penting dari kelas mayoritas, mengakibatkan model kurang mampu dalam menggeneralisasi pada situasi di luar data pelatihan. Oleh karena itu, pilihan oversampling dengan metode SMOTE dianggap lebih efektif untuk menangani ketidakseimbangan data dalam kasus deteksi penipuan pada transaksi kartu kredit.
