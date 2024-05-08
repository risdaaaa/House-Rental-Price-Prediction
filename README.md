
# Proyek Pertama House Rental Price Prediction - Dwi krisdanarti
## Domain Proyek
### Latar Belakang
Prediksi harga sewa rumah merupakan bagian penting dalam pemahaman fenomena pasar properti. Dalam beberapa tahun terakhir, kebutuhan akan prediksi harga sewa rumah telah semakin meningkat, seiring dengan pertumbuhan industri properti dan permintaan akan akomodasi yang terjangkau. Faktor-faktor seperti lokasi, ukuran properti, fasilitas yang tersedia, dan kondisi pasar secara umum memengaruhi harga sewa rumah. Namun, dengan volume data yang terus bertambah dan kompleksitas pasar yang berkembang, metode konvensional dalam menentukan harga sewa rumah mulai terasa kurang efisien. Inilah latar belakang pentingnya penggunaan teknik machine learning dalam memprediksi harga sewa rumah.

Dengan memanfaatkan algoritma dan model machine learning, kita dapat menggali pola-pola tersembunyi dalam data pasar properti yang besar dan kompleks. Hal ini memungkinkan penyusunan prediksi harga sewa yang lebih akurat dan responsif terhadap perubahan-perubahan pasar yang dinamis. Pembelajaran mesin membangun algoritma dan membangun model dari data, lalu menerapkannya pada data baru untuk dibuat prediksi. Regresi Linier, KNN, Jaringan Syaraf Tiruan, dan Pembelajaran Mendalam adalah beberapa pembelajaran mesin yang populer algoritma[1]. Dengan demikian, penerapan machine learning dalam prediksi harga sewa rumah tidak hanya memberikan keuntungan bagi pemilik properti dan penyewa, tetapi juga memberikan wawasan yang berharga bagi para pemangku kepentingan di bidang properti dan pasar finansial secara keseluruhan.

### Bussiness Understanding
Prediksi harga sewa rumah memiliki implikasi besar bagi investor, pemilik properti, dan calon penyewa. Akurasi prediksi harga sewa memungkinkan investor membuat keputusan investasi yang lebih cerdas, membantu pemilik properti menetapkan harga sewa yang kompetitif, dan memungkinkan calon penyewa untuk merencanakan keuangan dengan lebih baik. Selain itu, pemahaman tren harga sewa juga bermanfaat bagi lembaga keuangan dan pemerintah dalam perencanaan kebijakan perumahan dan pembangunan ekonomi yang berkelanjutan. Oleh karena itu, penggunaan teknik machine learning dalam prediksi harga sewa rumah memiliki dampak yang signifikan dalam berbagai aspek ekonomi dan sosial.

#### Problem Statement
- fitur apa yang paling berpengaruh terhadap harga sewa rumah?
- Metode apa yang paling akurat untuk prediksi harga rumah?

### Goals
- Mengetahui fitur yang paling berpengaruh pada harga sewa rumah.
- Mengetahui metode yang paling akurat untuk memprediksi harga rumah.


### Solution statements
- Untuk menganalisis data, dapat dilakukan dengan dua pendekatan utama: analisis univariat dan analisis multivariat. Analisis univariat fokus pada satu fitur pada satu waktu, sedangkan analisis multivariat mempertimbangkan hubungan antara beberapa fitur secara bersamaan. Visualisasi data juga dapat membantu dalam memahami pola dan hubungan antar fitur. Memahami data melalui analisis dan visualisasi membantu dalam mengidentifikasi korelasi antar fitur, yang penting untuk pemahaman yang lebih baik tentang dataset.
- Menemukan metode paling akurat untuk memprediksi harga rumah melibatkan evaluasi berbagai model menggunakan metrik yang sesuai seperti Mean Squared Error (MSE) atau R-squared. Model dengan nilai MSE yang lebih rendah atau nilai R-squared yang lebih tinggi dianggap lebih akurat dalam memprediksi harga rumah. Evaluasi ini dapat dilakukan melalui teknik validasi silang (cross-validation) atau pembagian dataset menjadi data latih dan data uji.

## Data Understanding
Data ini awalnya dikumpulkan oleh Austin Reese pada tanggal 7 Januari 2020 dari Craiglist.org. Sumber data ini bersifat publik.

Dataset yang digunakan [house-rent-prediction-dataset](https://www.kaggle.com/datasets/rkb0023/houserentpredictiondataset).

Informasi pada dataset:
- Dataset memiliki format CSV (Comma-Seperated Values).
- Dataset memiliki 265190 sample dan 22 fitur.
- Dataset memiliki 10 fitur bertipe int64, 9 fitur bertipe object, dan 3 fitur bertipe float 64.

### Variabel-variabel pada house rent prediction dataset adalah sebagai berikut:
Dataset ini terdiri dari beberapa kolom yang mencakup informasi terkait properti sewa, termasuk:

- **id**: Nomor identifikasi unik untuk setiap rumah dalam dataset.
- **url**: URL yang terkait dengan properti pada situs web atau platform tertentu.
- **region**: Wilayah geografis di mana properti tersebut terletak.
- **region_url**: URL yang terkait dengan wilayah geografis tempat properti tersebut berada.
- **price**: Harga sewa properti.
- **type**: Jenis properti, seperti apartemen, rumah, atau kondominium.
- **sqfeet**: Luas properti dalam satuan kaki persegi.
- **beds**: Jumlah kamar tidur di properti.
- **baths**: Jumlah kamar mandi di properti.
- **cats_allowed**: Indikator apakah kucing diizinkan di properti tersebut (biasanya bernilai 0 untuk tidak diizinkan dan 1 untuk diizinkan).
- **dogs_allowed**: Indikator apakah anjing diizinkan di properti tersebut (biasanya bernilai 0 untuk tidak diizinkan dan 1 untuk diizinkan).
- **smoking_allowed**: Indikator apakah merokok diizinkan di properti tersebut (biasanya bernilai 0 untuk tidak diizinkan dan 1 untuk diizinkan).
- **wheelchair_access**: Indikator apakah properti tersebut dapat diakses oleh kursi roda (biasanya bernilai 0 untuk tidak dapat diakses dan 1 untuk dapat diakses).
- **electric_vehicle_charge**: Indikator apakah properti tersebut menyediakan fasilitas pengisian kendaraan listrik (biasanya bernilai 0 untuk tidak tersedia dan 1 untuk tersedia).
- **comes_furnished**: Indikator apakah properti tersebut disewakan dengan perabotan (biasanya bernilai 0 untuk tidak furnished dan 1 untuk furnished).
- **laundry_options**: Opsi pencucian pakaian yang tersedia di properti, seperti mesin cuci di dalam unit atau di area bersama.
- **parking_options**: Opsi parkir yang tersedia di properti, seperti parkir di tempat atau parkir jalanan.
- **image_url**: URL gambar-gambar properti.
- **description**: Deskripsi properti yang mungkin mencakup detail-detail seperti fasilitas, lokasi, dan kondisi.
- **lat**: Koordinat lintang properti.
- **long**: Koordinat bujur properti.
- **state**: Negara bagian tempat properti tersebut terletak.

### Univariate Analysis
#### Categorical Features

![numerik-1](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/b6a6bb36-d4db-4b4f-9450-77ff071855d8)

Gambar 1. Grafik categorical features

Pada gambar 1, properti yang dikategorikan sebagai 'apartment' memiliki harga setara atau lebih besar dari $200,000. Sementara itu, properti dengan tipe 'house', 'townhouse', 'condo', 'duplex', 'manufactured', 'cottage/cabin', 'loft', 'flat', 'in-law', 'land', dan 'assisted living' memiliki harga kurang dari $50,000.
##### Numerical Features

![numerical](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/94344f9a-288c-4a64-9331-d1b4d50128b3)

Gambar 2. Grafik categorical feature

Pada gambar 2 terdapat 10 grafik yang memvisualisasikan persebaran dari tiap categorical feature.
### Multivariate Analysis
#### Categorical Features

![relatif-price](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/1c934afc-db9f-48a5-90a5-b00423101bef)

Gambar 3. Grafik harga berdasarkan fitur tipe

Berdasarkan grafik diatas, fitur type rumah memiliki dampak signifikan terhadap harga sewa rata-rata, terutama tipe apartemen.

#### Correlation Matrix

![correlation](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/4d81a89f-af34-446b-b009-2ea708f8e3a4)

Gambar 4. Visualisasi Ccorrelation matrix

Fitur 'cats_allowed' dan 'dogs_allowed' memiliki korelasi yang sangat tinggi dengan angka 0,89.

## Data Preparation
### One Hot Encoding
One-hot encoding adalah teknik yang mengubah variabel kategori menjadi representasi biner, di mana setiap nilai kategori direpresentasikan oleh satu kolom dengan nilai 1 atau 0. Fitur yang diubah pada proyek ini adlah 'type'.
### Reduksi Dimensi dengan PCA
Reduksi dimensi dengan PCA adalah teknik yang berguna untuk mengurangi kompleksitas data dengan memproyeksikan fitur-fitur asli ke dalam ruang dimensi yang lebih rendah. Misalnya, jika kita memiliki fitur "cats_allowed" dan "dogs_allowed" dalam dataset, kita dapat menggunakan PCA untuk menggabungkan kedua fitur ini menjadi satu dimensi baru yang mencerminkan variasi terbesar dalam data. Dengan demikian, kita dapat mengurangi dimensi dataset tanpa kehilangan informasi penting tentang kehadiran hewan peliharaan di properti.
### Train-Test-Split
  
![dataset](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/af36f1e8-fd08-4d23-9ac8-38d4ed44cffb)

Gambar 5. Jumlah sampel data

Jumlah data adalah 265190 sampel. Pada train test spit, data akan dibagi menjadi data train dan data test. Data train akan digunakan untuk membangun model, sedangkan data test akan digunakan untuk menguji performa model. Pada proyek ini data train sebesar 238671 dan data test sebesar 26519.

### Standarisasi
  
![standarisasi](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/c23bf8e9-8758-4dfe-8059-7ef863d0ea20)

|   | animals  | sqfeet  | beds  | baths  | smoking_allowed  | wheelchair_access  | electric_vehicle_charge  | comes_furnished  |
|---|---|---|---|---|---|---|---|---|
| count  | 238671.0  | 238671.0  | 238671.0  | 238671.0  | 238671.0  | 238671.0  | 238671.0  | 238671.0  |
| mean  |  0.0	 | -0.0	  | 0.0	  |  0.0	 |  0.0	 |  -0.0	 | -0.0	  | -0.0	  |
| std  |  1.0 | 1.0  | 1.0  | 1.0  |  1.0 | 1.0  |  1.0 | 1.0  |
| min  | -0.7  | -0.0  | -0.5  | -2.3  | -1.7  |  -0.3 |  -0.1 |  -0.2 |
| 25%  |  -0.7	 |  -0.0 | -0.2  |  -0.8 | -1.7  | -0.3  |  -0.1 | -0.2  |
| 50%  | -0.7  | -0.0  | 0.0  | -0.8	  | 0.6  | -0.3  | -0.1  | -0.2  |
| 75%  | 1.6  | 0.0  | 0.0  | 0.8  |  0.6 | -0.3  | -0.1  |  -0.2 |
| max  |  1.6 | 347.4  | 283.0  | 115.9  | 0.6  |  3.4 |  8.3 | 4.4  |

Gambar 6. Hasil standarisasi

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

## Modeling
### K-Nearest Neighbour
Dalam proses ini, digunakan model K-Nearest Neighbors Regressor (KNN) dengan parameter jumlah tetangga `(n_neighbors)` sebanyak 10. Model tersebut dilatih menggunakan data latih (X_train dan y_train), dan kemudian MSE dari prediksi terhadap data latih dihitung dan disimpan sebagai metrik evaluasi. 
Meskipun KNN sederhana dalam konsepnya, namun memiliki kelemahan pada data dengan dimensi yang besar. Ini terjadi ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi pada data.
### Random Forest
Mengimpor RandomForestRegressor dari library scikit-learn dan membuat model prediksi menggunakan algoritma Random Forest Regressor. Parameter-parameter yang digunakan adalah sebagai berikut:
-`n_estimators`: Jumlah pohon keputusan dalam model, diatur ke 50.
-`max_depth`: Kedalaman maksimum setiap pohon, diatur ke 16.
-`random_state`: Seed untuk mengontrol randomness dalam pembangunan model, diatur ke 55.
-`n_jobs`: Jumlah pekerjaan yang akan dijalankan paralel, diatur ke -1 untuk menggunakan semua core CPU yang tersedia.
Setelah model dibuat, dilakukan pelatihan menggunakan data latih (X_train dan y_train), dan nilai MSE dari prediksi terhadap data latih dihitung menggunakan fungsi mean_squared_error dan disimpan dalam dataframe `models` pada baris yang sesuai dengan model Random Forest dan kolom 'train_mse'.
### Boosting 
Mengimpor AdaBoostRegressor dari library scikit-learn dan membuat model prediksi menggunakan algoritma Boosting. Parameter-parameter yang digunakan adalah sebagai berikut:
-`learning_rate`: Tingkat pembelajaran untuk setiap estimator dalam ensemble, diatur ke 0.05.
-`random_state`: Seed untuk mengontrol randomness dalam pembangunan model, diatur ke 55.
Setelah model dibuat, dilakukan pelatihan menggunakan data latih (X_train dan y_train), dan nilai MSE dari prediksi terhadap data latih dihitung menggunakan fungsi mean_squared_error dan disimpan dalam dataframe `models` pada baris yang sesuai dengan model Boosting dan kolom 'train_mse'.

Dalam evaluasi model di atas, Mean Squared Error (MSE) digunakan sebagai metrik untuk mengevaluasi performa model pada data latih dan data uji. Semakin kecil nilai MSE, semakin baik performa model dalam memprediksi target.

Dari hasil evaluasi tersebut, dapat dilihat bahwa metode yang memberikan MSE terendah pada data uji adalah Random Forest (RF), diikuti oleh Boosting, dan yang terakhir adalah KNN. Hal ini menunjukkan bahwa Random Forest adalah yang paling akurat di antara ketiga metode tersebut dalam memprediksi target pada dataset yang digunakan.

## Evaluation
![MSE](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/861f490e-3582-4fc7-8112-da3a5f4ab0b0)

|           |  Train              |   Test             |
|-----------|---------------------|--------------------|
| KNN       | 27300919949.457352  | 960738.501214      |
| RF        | 22340979067.296822  | 458604.857434      | 
| Boosting  | 65426441569.944214  | 49779899023.228065 | 

Gmabar 7. MSE

Model Random Forest (RF) memiliki nilai MSE terendah pada data uji, yaitu sekitar 458,605. Model KNN memiliki nilai MSE sekitar 960,739, sedangkan model Boosting memiliki nilai MSE yang paling tinggi, yaitu sekitar 49,779,899,023.

![msee](https://github.com/risdaaaa/House-Rental-Price-Prediction/assets/147994396/d666a572-b187-4dd2-97a2-05ba71e8b0b7)


Gambar 8. Grafik MSE

Berdasarkan hasil tersebut, dapat disimpulkan bahwa model Random Forest (RF) adalah yang paling akurat di antara ketiga model tersebut dalam memprediksi target pada dataset yang digunakan.



## References
[1] Febriyanto, F. D., Endroyono, & Kusnendar, Y. (2023). House Price Prediction using Multiple Linear. JAREE (Journal on Advanced Research in Electrical Engineering), 7.

