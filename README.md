# Credit_Card_Fraud
Credit card fraud adalah tindakan atau usaha untuk mendapatkan keuntungan secara tidak sah dengan menggunakan informasi kartu kredit orang lain tanpa izin. Ini dapat melibatkan aktivitas seperti pembelian barang atau layanan tanpa pengetahuan atau persetujuan pemilik kartu, atau bahkan mencuri informasi kartu kredit untuk melakukan transaksi ilegal. 

Dalam proyek ini, saya akan membangun model machine learning yang dapat mengidentifikasi apakah suatu transaksi dengan kartu kredit bersifat penipuan atau tidak menggunakan beberapa algoritma dan teknik klasifikasi. Tujuan utama proyek ini adalah menciptakan model yang dapat membantu mengamankan transaksi keuangan dengan mengenali pola-pola yang terkait dengan aktivitas penipuan pada kartu kredit.

## Project Outline
1. Dataset
2. Data Analysis
3. Model Building
4. Model Evaluation

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
