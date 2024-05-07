# -*- coding: utf-8 -*-
"""House Rental Price Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14GJ8wZ5mEy5iZo0h-3GlAtA41anJOzMd

Data Collection
"""

from google.colab import files
files.upload()

! mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d rkb0023/houserentpredictiondataset

!unzip /content/houserentpredictiondataset.zip

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

df = pd.read_csv("/content/houseRent/housing_train.csv")
df.head()

df

"""Data Understanding & Removing Outlier"""

df.shape

df.info

df.describe()

"""EDA - Deskripsi variabel"""

df.info()

df.describe()

"""Missing value"""

missing_values = df.isnull().sum()
print(missing_values)

cleaned_df = df.dropna(axis=1)

# Mengecek jumlah nilai yang hilang setelah penghapusan
print(cleaned_df.isnull().sum())

# Menghapus kolom "id", "url", dan "region_url"
cleaned_df = cleaned_df.drop(columns=["id", "url", "region_url", "region", "image_url"])

# Menampilkan informasi DataFrame setelah menghapus kolom
print(cleaned_df.info())

cleaned_df.shape

cleaned_df.info()

cleaned_df

"""Univariate Analysis"""

cleaned_df.describe()

numerical_features = ['price', 'sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished']
categorical_features = ['type']

"""Categorical Features"""

feature = categorical_features[0]
count = cleaned_df[feature].value_counts()
percent = 100*cleaned_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(cleaned_df)
count.plot(kind='bar', title=feature);

"""Numerical Features"""

cleaned_df.hist(bins=50, figsize=(20,15))
plt.show()

"""Exploratory Data Analysis - Multivariate Analysis

Categorical Features
"""

cat_features = cleaned_df.select_dtypes(include='object').columns.to_list()

for col in cat_features:
  sns.catplot(x=col, y="price", kind="bar", dodge=False, height = 4, aspect = 3,  data=cleaned_df, palette="Set3")
  plt.title("Rata-rata 'price' Relatif terhadap - {}".format(col))

"""Numerical Features"""

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(cleaned_df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
# Mengambil hanya fitur numerik dari DataFrame
# numeric_features = cleaned_df.select_dtypes(include=['int64', 'float64'])

# Menghitung korelasi matriks untuk fitur numerik
correlation_matrix = numeric_features.corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""Data Preparation

Encoding Fitur Kategori
"""

from sklearn.preprocessing import  OneHotEncoder
cleaned_df = pd.concat([cleaned_df, pd.get_dummies(cleaned_df['type'], prefix='type')],axis=1)
cleaned_df.drop(['type'], axis=1, inplace=True)
cleaned_df.head()

"""Reduksi Dimensi dengan PCA"""

sns.pairplot(cleaned_df[['cats_allowed','dogs_allowed']], plot_kws={"s": 3});

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=123)
pca.fit(cleaned_df[['cats_allowed','dogs_allowed']])
princ_comp = pca.transform(cleaned_df[['cats_allowed','dogs_allowed']])

pca.explained_variance_ratio_.round(2)

import pandas as pd
from sklearn.decomposition import PCA

# Melakukan PCA dengan 1 komponen
pca = PCA(n_components=1, random_state=123)
pca_result = pca.fit_transform(cleaned_df[['cats_allowed', 'dogs_allowed']])

# Membuat DataFrame baru dari hasil transformasi PCA
pca_df = pd.DataFrame(data=pca_result, columns=['animals'], index=cleaned_df.index)

# Menggabungkan DataFrame baru dengan DataFrame asli
cleaned_df = pd.concat([cleaned_df, pca_df], axis=1)

# Menghapus kolom-kolom asli
cleaned_df.drop(['cats_allowed', 'dogs_allowed'], axis=1, inplace=True)

"""Train-Test-Split"""

from sklearn.model_selection import train_test_split

X = cleaned_df.drop(["price"],axis =1)
y = cleaned_df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""Standarisasi"""

from sklearn.preprocessing import StandardScaler

numerical_features = ['animals', 'sqfeet', 'beds', 'baths', 'smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(1)

"""Model Development

Model Development dengan K-Nearest Neighbor
"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""Model Development dengan Random Forest"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor

# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""Model Development dengan Boosting Algorithm"""

from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""Evaluasi Model"""

from sklearn.preprocessing import StandardScaler

# Mendefinisikan StandardScaler
scaler = StandardScaler()

# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)