# Proyek Pertama House Rental Price Prediction - Dwi krisdanarti
## Domain Proyek
### Latar Belakang
Prediksi harga sewa rumah merupakan bagian penting dalam pemahaman fenomena pasar properti. Dalam beberapa tahun terakhir, kebutuhan akan prediksi harga sewa rumah telah semakin meningkat, seiring dengan pertumbuhan industri properti dan permintaan akan akomodasi yang terjangkau. Faktor-faktor seperti lokasi, ukuran properti, fasilitas yang tersedia, dan kondisi pasar secara umum memengaruhi harga sewa rumah. Namun, dengan volume data yang terus bertambah dan kompleksitas pasar yang berkembang, metode konvensional dalam menentukan harga sewa rumah mulai terasa kurang efisien. Inilah latar belakang pentingnya penggunaan teknik machine learning dalam memprediksi harga sewa rumah. Dengan memanfaatkan algoritma dan model machine learning, kita dapat menggali pola-pola tersembunyi dalam data pasar properti yang besar dan kompleks. Hal ini memungkinkan penyusunan prediksi harga sewa yang lebih akurat dan responsif terhadap perubahan-perubahan pasar yang dinamis. Dengan demikian, penerapan machine learning dalam prediksi harga sewa rumah tidak hanya memberikan keuntungan bagi pemilik properti dan penyewa, tetapi juga memberikan wawasan yang berharga bagi para pemangku kepentingan di bidang properti dan pasar finansial secara keseluruhan.

Referensi : [House Price Prediction using Multiple Linear Regression and KNN](http://jaree.its.ac.id/index.php/jaree/article/view/328)

## Bussiness Understanding
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
Dataset yang digunakan [house-rent-prediction-dataset](https://www.kaggle.com/datasets/rkb0023/houserentpredictiondataset)
Informasi pada dataset:
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
##### Categorical Features
![](https://previews.dropbox.com/p/thumb/ACQHDMVk7PQG-Vx5edufJ-9AW1lnJI2rQeZf4hhTv_v5-tsYshZuaCHKopoJqyhyUrHqXGoRtSHHsv1n8k6Mrmu8r7junl1wGgd4gwplnqlUKCadQJ8eRX8RRYNcWTF0lPePwUx9MCedCTrFwbjIwmjvTViVU5AJfIOB6cwVXTxxof4EiybLrNnkUzVAn8M2gsiChn2VFebM2olZ3zVPXX_nrecyCF2RFe1WbGg_vq6_EIrv2EEazgoVixRGGazTxlc-obpDHylDaueR-1zfAX9JCjp6k7G-6PmcnB6NK_g53VjFB4cg_xxNtSp5V80_Esm1d20Fc6c4hzCsfny1zInd/p.png)
- Properti yang dikategorikan sebagai 'apartment' memiliki harga setara atau lebih besar dari $200,000. Sementara itu, properti dengan tipe 'house', 'townhouse', 'condo', 'duplex', 'manufactured', 'cottage/cabin', 'loft', 'flat', 'in-law', 'land', dan 'assisted living' memiliki harga kurang dari $50,000.
##### Numerical Features
![](https://previews.dropbox.com/p/thumb/ACRAIQEbCSsSNPl-GYZibtbtOkmmS15cSs_9bfxmV-PtR08c4xYHpoSjuvczv1HHZutd-Q5lq7TpLNWhsX3xzHAG7t8hPzRgzZKi-BllELX6WYsMAEIgfq6PSlc7BGxBdGkNB-YkNyifLY-0apqNth20YPzMkdE9UOWUni66BGYSTV2p_w7giiII38_eXr_bJ5pOcBqQHkEUdv2xN6LO0JggenFLtbShnI8i4iJZiDaD5CSjpDp0A2k3O2YC1nWZR3DL5JJbAwSIdWHtwOFENy1seQqaXxsUeFwkAquiy22ueKfw2hjOSqVfcR5NzXJ86MLyUAEzR0CtyDkD5jQl6HdE/p.png)

### Multivariate Analysis
##### Categorical Features
![](https://previews.dropbox.com/p/thumb/ACSpB-lxARytarlVyugv2EDQxyN167Uhf6BJuW2IKTYYNcrAl4Fa3geVOtRjxHtwg2rEqbxlN2mqU9Du96G7gK22hyTbSMBXIbBxldfw9PJDFU2jGSE2M1D-YjO7SOANawd9xEU5TdWSuoUsFbIhAe73_Rq9GQlBWiYGOwCyUvPianyQ8VtUCsHYhsNUkUWIg2VFtdZe9U-upoLxylhTWR0ep0N_V76i0uCIcCRagTYXwWxfByijX6VI4q4fgbudgq1FsZ5odm2S1e0FZJdhV-bTH50Tpb6T0iQmUY1CE1SbK7cilT-7y2vXs8h194IlILacPT_8TvlC9yk_R-MzrdEJ/p.png)
- Fitur type rumah memiliki dampak signifikan terhadap harga sewa rata-rata, terutama ketika tipe propertinya adalah apartemen.
##### Numerical Features
![](https://previews.dropbox.com/p/thumb/ACRo87QcuDYh1Tk2yZ_vQ1DAoD4TVSe1mBQW5ruplDgzYFTiE_jnXGfs70-hgqo36bpgflwz1QBc0t8LwKT_0PDiT0p9wsVKDEkjPS8jY055yKs0Xl5q3W7bRtFyewiYc4QqKvbx0_BOGBCJZtvmrKsNVo_90HGJ8fz3fuwMxGInKo3ml_nd6rWSKtM6jg0WKSqz1HOO4PKI82MFCiNW_SklSqdBZjDeEh6n4CQz3ARD90zHh_dvKsRcKEWVC71I0mXqMFGVJDbg6mqaF0shkilY-ksXImbY5ygwh_RlG3dlVwAyPj6tnSo5csQuf2KPdES9b7NOMto5voYRsXR5rgfR/p.png)
##### Correlation Matrix
![](https://previews.dropbox.com/p/thumb/ACS0khxLpIpz2NTufKvRWqiu3PxfjROwkZx2olXxIXnEor-Y90RC86Xm6MUD-txrVvmRq0W9vuM8P7TmQqMUDFSYnR3BdDVHGzfBeKMKBa-xwKsyyPNUAhHJuv_Zx_Ja5Iw-H12zqeOi4i8-i3e5UUDNTNAWALbXzgK6mAPM5WDgywJfoyPs0NcO41lD7_OzFwqV0Ii6ni8yrDaHzfl3plXmay08NyaAD5KlzQP5xeM1MNizG6ECz6y93_yuK6r96k6DJGwpB-3aT8hGBeLf-86Au3rAVYIzuR4U8_zfIg7BdHk6V6Or0cPjilQvvFG1-W4szhxSwVmNtoAghxikBm7Q/p.png)
- Fitur 'cats_allowed' dan 'dogs_allowed' memiliki korelasi yang sangat tinggi dengan angka 0,89.

## Data Preparation
- One Hot Encoding
One-hot encoding adalah teknik yang mengubah variabel kategori menjadi representasi biner, di mana setiap nilai kategori direpresentasikan oleh satu kolom dengan nilai 1 atau 0. Fitur yang diubah pada proyek ini adlah 'type'.
- Reduksi Dimensi dengan PCA
Reduksi dimensi dengan PCA adalah teknik yang berguna untuk mengurangi kompleksitas data dengan memproyeksikan fitur-fitur asli ke dalam ruang dimensi yang lebih rendah. Misalnya, jika kita memiliki fitur "cats_allowed" dan "dogs_allowed" dalam dataset, kita dapat menggunakan PCA untuk menggabungkan kedua fitur ini menjadi satu dimensi baru yang mencerminkan variasi terbesar dalam data. Dengan demikian, kita dapat mengurangi dimensi dataset tanpa kehilangan informasi penting tentang kehadiran hewan peliharaan di properti.
- Train-Test-Split
![](https://previews.dropbox.com/p/thumb/ACTSH1FQuFtOLTx1JHkTWaS0jY72_u08nPwsL2OLxDlZPtAITSCuU0m0qeUnBpB8lshSiRKCgw9JSe63b_i2daza8-eHLGGLg6goUomiRfl9grKlM71mU4ZFCiS1EmmBqp_P_msrVOLUuL-YT2H5BvYFcpLtmXr9PzM28Qe5maCpeVDOG339bcen09HHvOcx8RCGCJkF0oU2qg0Bxy15zWN2BIKgx3vEmSta8oWfS7w1lc8f--mde7zkzz0sw-AZXEsBJa3Fo_oTJbhZYlcb351dBnL-rQxnvz5MtYfErjvLcxYL9juPhhobPiZpSL8nPwgzWp8Uea371K88G3xj5meX/p.png)
Jumlah data adalah 265190 sampel. Pada train test spit, data akan dibagi menjadi data train dan data test. Data train akan digunakan untuk membangun model, sedangkan data test akan digunakan untuk menguji performa model. Pada proyek ini data train sebesar 238671 dan data test sebesar 26519.

- Standarisasi
![](https://previews.dropbox.com/p/thumb/ACSPdPhdAfO4qbavSaqRGM1DBCI0WOf9dkpvC_mTDpKjekZcIl4YH6NMiOWhkK20HX9kw4fFbQP6QmKwvohM4dzJJb95a7vIB0Tk_v1zU0O45UXTj5t4sdlKws0fasN1o13yNU7iDD_AiM632b1ptVJsb0UdOWu8dfTJlLxFpzGlYkSdv3IJxlCYEFv8gE9uH89sbFDA5_O5ecRLbA4lkFHaIT4BZLRmsoPkSPwwT2yjQPCAbmZ0U47NFUeX03zCou-FUSOeCa-V8sge6gMs21ryZekHK6lgEsS-k2xxeiZVb__cnF18kKBwmENc_33H3wUchsM9d71SbWqp1Ynm8wa3/p.png)
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

## Modeling
- K-Nearest Neighbour
Dalam proses ini, digunakan model K-Nearest Neighbors Regressor (KNN) dengan parameter jumlah tetangga `(n_neighbors)` sebanyak 10. Model tersebut dilatih menggunakan data latih (X_train dan y_train), dan kemudian MSE dari prediksi terhadap data latih dihitung dan disimpan sebagai metrik evaluasi. 
Meskipun KNN sederhana dalam konsepnya, namun memiliki kelemahan pada data dengan dimensi yang besar. Ini terjadi ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi pada data.
- Random Forest
Mengimpor RandomForestRegressor dari library scikit-learn dan membuat model prediksi menggunakan algoritma Random Forest Regressor. Parameter-parameter yang digunakan adalah sebagai berikut:
-`n_estimators`: Jumlah pohon keputusan dalam model, diatur ke 50.
-`max_depth`: Kedalaman maksimum setiap pohon, diatur ke 16.
-`random_state`: Seed untuk mengontrol randomness dalam pembangunan model, diatur ke 55.
-`n_jobs`: Jumlah pekerjaan yang akan dijalankan paralel, diatur ke -1 untuk menggunakan semua core CPU yang tersedia.
Setelah model dibuat, dilakukan pelatihan menggunakan data latih (X_train dan y_train), dan nilai MSE dari prediksi terhadap data latih dihitung menggunakan fungsi mean_squared_error dan disimpan dalam dataframe `models` pada baris yang sesuai dengan model Random Forest dan kolom 'train_mse'.
- Boosting 
Kode di atas mengimpor AdaBoostRegressor dari library scikit-learn dan membuat model prediksi menggunakan algoritma Boosting. Parameter-parameter yang digunakan adalah sebagai berikut:
-`learning_rate`: Tingkat pembelajaran untuk setiap estimator dalam ensemble, diatur ke 0.05.
-`random_state`: Seed untuk mengontrol randomness dalam pembangunan model, diatur ke 55.
Setelah model dibuat, dilakukan pelatihan menggunakan data latih (X_train dan y_train), dan nilai MSE dari prediksi terhadap data latih dihitung menggunakan fungsi mean_squared_error dan disimpan dalam dataframe `models` pada baris yang sesuai dengan model Boosting dan kolom 'train_mse'.

Dalam evaluasi model di atas, kita menggunakan Mean Squared Error (MSE) sebagai metrik untuk mengevaluasi performa model pada data latih dan data uji. Semakin kecil nilai MSE, semakin baik performa model dalam memprediksi target.

Dari hasil evaluasi tersebut, kita dapat melihat bahwa metode yang memberikan MSE terendah pada data uji adalah Random Forest (RF), diikuti oleh Boosting, dan yang terakhir adalah KNN. Hal ini menunjukkan bahwa Random Forest adalah yang paling akurat di antara ketiga metode tersebut dalam memprediksi target pada dataset yang digunakan.

## Evaluation

![](https://previews.dropbox.com/p/thumb/ACRBbaTZxQU36rwZki9fP9qOljrYj6pUMrWzMee0ZIJ_Jg_gJQBUtK8rTiSei3Syp_FPaitnMzXKRCejaVx1AbZWcbuZ-1IM2xuDEtNIZRDwqmXx14m1k3rv4kSzlrsEE6D3MmjJZT3gz01VoGbMunH5l6e92wD5-LGPfmuyUK8bTB5s3iTPlagzm95Srt8CKstcWcYwKcIuyuk1-4ZFEscHy9Flr3fJmO3ittdkH3hCmG1Rn_OpIcO2_YKr18diqIyRNWhOX5RQD-sahNbJMIXSN8UEe5kMAOLNSCkrUXD9iNFdv0HAQAap2bXnRXRXYQHKsP2lE6apNTr7MyGAqXEe/p.png)

Model Random Forest (RF) memiliki nilai MSE terendah pada data uji, yaitu sekitar 458,605. Model KNN memiliki nilai MSE sekitar 960,739, sedangkan model Boosting memiliki nilai MSE yang paling tinggi, yaitu sekitar 49,779,899,023.

![](https://previews.dropbox.com/p/thumb/ACTp_RwzKrMrEW5MSh2XHJb6jP4BbcReFHP3qeafVRfN3ucQxuvyHgQKumwCw3nuNXgH0931eWSXlGgvwo40pYZG7lOmvDgoCJGqpnXJzOsQumE4k4AOd4Tx1BQ6zuSWuqCbOb0Mj3KOdc76MocEr-bYrZt8cO0zP_-5ZW8TC5QBbMHDrNdua7Zbs2lnfoipV5oPPSxSUNe9Dde3VkZ_j83KKc9TTVY32JoErVDH1lP9E68BNY0_HID0POhawpWs8rH123mmtGIiOBC5wKSJY0y1-2M2j_g3tdiUtFuzfB0Z6V-T5BbgxvdyUjsrgWu3cvROiwTfnyIFb4sKHxce7KOY/p.png)

Berdasarkan hasil tersebut, dapat disimpulkan bahwa model Random Forest (RF) adalah yang paling akurat di antara ketiga model tersebut dalam memprediksi target pada dataset yang digunakan.
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.
