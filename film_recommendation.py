# -*- coding: utf-8 -*-
"""AYOSARAHJANGANMALES.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eMzgIkISsM8mNns5dl-VvKSt_kRpOBR6

#<center> Profil Dicoding </center>

Nama  : Sarah Salsabila

Email : m314v4331@dicoding.org

Alamat: Karawang , Jawa barat .

##<center> System Recommendation </center>

## proyek Overview
Pada proyek kali ini, saya akan mencoba membuat sebuah sistem rekomendasi mengenai film .menurut wikipedia film merupakan serangkaian gambar diam, yang ketika ditampilkan pada layar akan menciptakan ilusi gambar bergerak karena efek fenomena phi. Ilusi optik ini memaksa penonton untuk melihat gerakan berkelanjutan antar objek yang berbeda secara cepat dan berturut-turut.

Menonton film pada masa pandemi menjadi salah satu hiburan masyarakat untuk menemani keseharian dikala karantina dan aktivitas yang mengharuskan untuk selalu di rumah saja.Seperti pemberitahuan dari beberapa media, diantaranya kompas.com dan indozone.com pada tahun 2020 , pengguna netflix meningkat sebesar 15,8 juta selama pandemi . netflix merupakan salah satu platform online untuk menonton film . dari artikel tersebut dapat diambil kesimpulan bahwa tingkat minat dalam menonton film selama pandemi meningkat baik melalui platform netflix atau platform lainnya . 

Saat akan menonton film , seseorang pasti akan merasa kebingungan untuk memilih film apa yang akan di tonton .Karena seperti yang kita tahu film memiliki banyak sekali genre dan juga judul . 

Oleh sebab itu , saya di sini akan mencoba membuat sistem rekomendasi film untuk membantu melakukan decision dan memberikan saran kepada seseorang mengenai film yang akan di tonton .

Berikut beberapa artikel dan refrensi mengenai Film dan sistem rekomedasi film 
- [kompas, Selama Pandemi Corona, Pengguna Baru Netflix Bertambah 15,8 Juta](https://www.kompas.com/hype/read/2020/05/11/160952866/selama-pandemi-corona-pengguna-baru-netflix-bertambah-158-juta)
- [film](https://id.wikipedia.org/wiki/Film)

## Business Understanding
Berdasarkan overview proyek di atas mengenai film , sebagai salah satu penikmat film saya juga merasa membutuhkan suatu rekomendasi untuk memilih film . 
Berdasarkan article yang saya lihat pada [kaskus](https://www.kaskus.co.id/thread/59e86f6aded770d8618b4568/bingung-nonton-film-apa-inilah-tip-memilih-film-berkualitas/) terdapat beberapa faktor yang dapat membantu kita untuk memilih film yang berkualitas atau kita sukai. Seperti ,
1. Genre
2. Trailer
3. Cast
4. Sutradara
5. Crew produksi
6. Komunitas pecinta film.

### Problem Statements
Dari artikel itulah saya mendapatkan beberapa pertanyaan ,seperti ,
1. Apa ada suatu cara untuk membantu menyarankan film kepada seseorang hanya dengan 1 faktor , semisal dari genre (karena seseorang biasanya menonton berdasarkan genre yang dia sukai ) ?
2. Apakah ada suatu sistem yang dapat membantu menyarankan film , berdasarkan film - film yang dilihat sebelumnya?
3. Dan dapatkah sistem tersebut memberikan rekomendasi lain berdasarkan penilaian dari pengguna lain .

### Goals
Untuk  menjawab pertanyaan tersebut, saya mencoba membuat predictive analisis dengan tujuan atau goals sebagai berikut:
- Membuat sebuah sistem yang dapat membantu seseorang untuk menonton film lain berdasarkan faktor genre .
- Membuat sebuah sistem yang dapat memberikan saran film berdasarkan film yang sebelumnya pernah dilihat .
- membuat sistem yang dapat memberikan saran film berdasarkan histori pengguna yang pernah memberikan penilaian pada suatu film .

### Solution approach
Untuk mencapai tujuan serta dapat menyelesaikan permasalahan pada problem statement , Saya akan menggunakan 2 Model filtering :
- **Content Based Filtering** 
 Ide dari sistem rekomendasi berbasis konten (content-based filtering) adalah merekomendasikan item yang mirip dengan item yang disukai pengguna pada masa lalu.
 Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai pada masa lalu atau sedang dilihat pada masa kini kepada pengguna. makin banyak informasi yang diberikan pengguna, makin baik akurasi sistem rekomendasi.
 Nantinya pada model ini saya akan mencoba memberikan rekomendasi berdasarkan film yang sebelumnya dilihat dan menggunakan faktor genre sebagai acuan rekomendasinya .
 
- **Collaborative Filtering**
 Teknik ini merekomendasikan item yang mirip dengan preferensi pengguna pada masa lalu. Teknik ini membutuhkan data rating dari user untuk menghasilkan rekomendasi sejumlah film yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna, kita akan mengidentifikasi film -film yang mirip dan belum pernah dilihat oleh pengguna untuk direkomendasikan.

Untuk hal pertama yang saya lakukan adalah melakukan install kaggle dengan library pip .
"""

!pip install -q kaggle
print('done!')

"""Selanjutnya melakukan import file pada google colab dengan menginputkan data API akun kaggle saya ."""

from google.colab import files
files.upload()

"""membuat direktori baru untuk menyimpan install data kaggle.json"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

"""Mengunduh dataset yang akan digunakan , Pada proyek machine learning ini saya menggunakan dataset mengenai daftar film untuk sistem rekomendasi . yaitu Movie Recommendation Data yang dapat diakses pada link berikut [movie recommendation data](https://www.kaggle.com/rohan4050/movie-recommendation-data)


Berikut baris code yang harus dilakukan 
"""

!kaggle datasets download -d rohan4050/movie-recommendation-data

"""Setelah mengunduh data ,langkah berikutnya adalah melakukan unzip pada data .
disini saya menyimpan data yang sudah di unzip pada berkas d dengan nama folder movie - recommendation .

dengan menggunakan baris kode sebagai berikut :
"""

!unzip movie-recommendation-data.zip -d movie_recommendation

"""###<center> Data Understanding </center>

Pada proyek machine learning ini saya menggunakan dataset mengenai daftar film untuk sistem rekomendasi . yaitu Movie Recommendation Data yang dapat diakses pada link berikut
[movie recommendation data](https://www.kaggle.com/rohan4050/movie-recommendation-data)
Setelah di unzip terlihat bahwa didalam file movie-recommendation data terdapat 5 buah berkas csv . diantaranya :
- links.csv
- movies.csv
- ratings.csv
- tags.csv
- movies_metadata.csv

Pada proyek machine learning kali ini saya hanya menggunakan 3 file csv saja , yaitu tags,rating dan movies . Oleh sebab itu selanjutnya adalah tahap membaca data atau data loading . 

lakukan import library pandas , karena kita akan menggunakan fungsi read_csv yang terdapat didalam library pandas .
untuk file movies.csv saya definisikan sebagai dataframe movie , lalu tags.csv sebagai dataframe tag dan untuk ratings.csv sebagai dataframe rating.

untuk baris kodenya sebagai berikut :
"""

import pandas as pd

movie = pd.read_csv('/content/movie_recommendation/ml-latest-small/movies.csv')
tags = pd.read_csv('/content/movie_recommendation/ml-latest-small/tags.csv')
rating = pd.read_csv('/content/movie_recommendation/ml-latest-small/ratings.csv')

print('okay done sar')

"""selanjutnya saya memanggil fungsi columns untuk melihat apa saja nama tabel pada setiap data ."""

print(' Kolom pada data Movie adalah : ', (movie.columns))
print(' Kolom pada data Tags adalah : ', (tags.columns))
print(' Kolom pada data rating adalah : ', (rating.columns))

"""Informasi yang dapat diambil dari hasil ekesekusi kode tersebut adalah 

- Dataframe movie memiliki 3 buah kolom yang terdiri dari ['movieId', 'title', 'genres'] .
- Dataframe tags memiliki 4 buah kolom terdiri dari ['userId', 'movieId', 'tag', 'timestamp'].
- Dataframe rating memiliki 4 buah kolom juga yaitu ['userId', 'movieId', 'rating', 'timestamp'].

berikutnya adalah mencoba mengenali data lebih dalam dengan melakukan pengecekan jumlah data, baris dan kolom pada setiap file .
"""

print('Jumlah data pada movie   : ',(movie.shape))
print('Jumlah data pada tags    : ',(tags.shape))
print('Jumlah data pada rating  : ',(rating.shape))
print('Jumlah data kolom movieId pada Movie   : ',len(movie.movieId.unique()))
print('Jumlah data kolom title pada Movie     : ',len(movie.title.unique()))
print('Jumlah data kolom genres pada Movie    : ',len(movie.genres.unique()))
print('Jumlah data kolom movieId pada tags    : ',len(tags.movieId.unique()))
print('Jumlah data kolom userId pada tags     : ',len(tags.userId.unique()))
print('Jumlah data kolom tag pada tags        : ',len(tags.tag.unique()))

"""Seperti yang terlihat bahwasannya dataframe movie memiliki 9742 baris dan 3 kolom , tags memiliki 3683 baris dan 4 kolom , dan rating dengan 100836 baris dengan 4 kolom .

langkah berikutnya adalah pengecekan apakah data memiliki nilai null , duplicat dan mencari tahu informasi lebih dalam tentang data melalui fungsi describe atau info .

Melihat apakah dataframe movie memiliki nilai yang berduplikat.
"""

movie.duplicated().sum()

"""melihat apakah dataframe movie memiliki nilai null ."""

movie.isnull().sum()

"""memanggil fungsi info() untuk mengetahui macam-macam tipe data pada dataframe movie"""

movie.info()

"""Melihat nilai pada kolom genres, apa saja genre yang masuk pada kolom tersebut ."""

print('Jenis genre yang ada : ',movie.genres.unique())

"""Dari hasil eksekusi didapatkan informasi sebagai berikut :

Pada dataframe movie 
- Tidak terdapat nilai yang berduplikat
- Tidak memiliki nilai null
- Movie memiliki 2 tipe data yaitu 2 bertipe data object dan 1 bertipe data int
- Movie memiliki berbagai macam kombinasi genre

Selanjutnya adalah melakukan pengecekan lebih dalam terhadap dataframe tags.

Memanggil fungsi info() untuk melihat toipe data dan informasi lainnya pada tags.
"""

tags.info()

"""Mengecek apakah dataframe tags memiliki nilai yang berduplikat atau tidak"""

tags.duplicated().sum()

"""Mengecek apakah dataframe tags memiliki nilai yang null atau tidak"""

tags.isnull().sum()

"""memanggil fungsi head() dengan parameter 1"""

tags.head(1)

"""menghapus kolom timestamp pada dataframe , karena tidak akan digunakan .

gunakan fungsi drop() dengan baris kode sebagai berikut :
"""

tags = tags.drop(columns=['timestamp'])
tags.head()

"""Dari hasil eksekusi kode - kode diatas , didapatkan informasi sebagai berikut :


Pada dataframe tags
- Tidak terdapat nilai yang berduplikat
- Tidak memiliki nilai null
- tags memiliki 2 tipe data yaitu 1 bertipe data object dan 3 bertipe data int
- tags sekarang memiliki 3 kolom karena kolom timestamp telah dihapus .

melakukan pengecekan mendalam terhadap dataframe rating.

pertama saya memanggil fungsi head() untuk melihat data
"""

rating.head()

"""memanggil fungsi describe untuk menegetahui minimal dan maksimal nilai pada dataframe rating lalu mean dan standar deviasinya."""

rating.describe()

"""Menghapus kolom timestamp pada dataframe rating karena tidak akan di gunakan ."""

rating = rating.drop(columns=['timestamp'])
rating.head(1)

"""Melihat jumlah nilai pada setiap kolom dengan menggunakan fungsi len () . unique() digunakan untuk mencari nilai yang unik ."""

print('Jumlah userID  :',len(rating.userId.unique()))
print('Jumlah movieId :',len(rating.movieId.unique()))
print('Jumlah data rating :',len(rating))

"""Di dapatkan informasi seperti ini :

- Dataframe rating sekarang memiliki 3 kolom , karena kolom timestamp dihapus
- terdapat 610 jumlah data pada kolom userId
- tedapat 9724 jumlah data pada kolom movieId
- terdapat 100836 jumlah data pada rating

##<center> Data Preprocessing </center>

Melakukan upaya penggabungan dataframe dengan menggunakan fungsi concatenate pada library numpy .

Pada dataframe ini yang akan digabungkan pertama adalah tags dan movie , dimana kita menggunankan movieId yang unik sebagai acuan dalam penggabungan ini agar data yang digabungkan dapat sinkron satu sama lain .
"""

import numpy as np

Movie_all = np.concatenate((
    movie.movieId.unique(),
    tags.movieId.unique(),
))

# Mengurutkan data dan menghapus data yang sama
Movie_all = np.sort(np.unique(Movie_all))

print('Jumlah seluruh data film berdasarkan movieId: ', len(Movie_all))

"""Dari hasil eksekusi di dapatkan informasi bahwa file pada kategori film  ketika digabung memiliki 9742 file movie yang unik.

selanjutnya adalah menggabungkan data dengan kategori user , disini saya menggunakan userId yang unik sebagai acuan kunci dalam penggabungan . dan file yang akan digabungkan adalah dataframe tags dan rating .

masih menggunakan fungsi concatenate dari library numpy berikut baris kode yang harus dijalankan :
"""

user_all = np.concatenate((
    rating.userId.unique(),
    tags.userId.unique(),
))

# Mengurutkan data dan menghapus data yang sama
user_all = np.sort(np.unique(user_all))

print('Jumlah seluruh data film berdasarkan userId: ', len(user_all))

"""Seperti yang terlihat pada hasil eksekusi sebelumnya bahwa hanya terdapat 610 data pengguna dari 8742 film yang memiliki rating .


Selanjutnya adalah usaha mengetahui jumlah rating . dengan cara menggabungkan terlebih dahulu kolom - kolom yang di butuhkan . 

- pertama adalah membuat definisi baru , disini saya membuat movieinfo dimana di dalamnya saya mengabungkan ketiga dataframe ,yaitu movie,tags,rating . 
- lalu menggabungkan rating dengan movieinfo berdasarkan movieId  , dengan fungsi merge 
"""

# Menggabungkan file title,genre,tags,userId ke dalam dataframe movieinfo 
movieinfo = pd.concat([movie,tags,rating])

# Menggabungkan dataframe rating dengan movieinfo berdasarkan nilai movieId
film = pd.merge(rating, movieinfo , on=['movieId'], how='left')
film

"""Setelah itu dilakukan pengecekan pada dataframe yang telah digabung (film) apakah data memiliki nilai null """

# Cek missing value dengan fungsi isnull()
film.isnull().sum()

"""Dari kedua proses eksekusi diatas didapatkan informasi bahwa 
- Terdapat nilai null pada beberapa kolom , kolom yang bersih dari nilai null adalah userId_x , movieId, dan rating_x.


langkah selanjutnya adalah menghitung jumlah rating , lalu melakukan penggabungan dengan fungsi groupby berdasarkan movieId ,dengan kode berikut :
"""

# Menghitung jumlah rating kemudian menggabungkannya berdasarkan movieId
film.groupby('movieId').sum()

"""Selanjutnya adalah membuat definisi all_film_rate dengan variabel rating yang sudah diketahui sebelumnya ."""

all_film_rate = rating
all_film_rate

"""Menggabungkan data film pada dataframe movie dengan all_film_rate , menggunakan fungsi merge ."""

all_film_name = pd.merge(all_film_rate, movie[['movieId','title','genres']], on='movieId', how='left')
all_film_name

"""Setelah mengabungkan fitur mengenai film dan genre serta ratingnya selanjutnya adalah menggabungnya dengan dataframe tags untuk mendapatkan userId dan tag label apa saja yang diberikan user pada film - film tersebut .masih dengan menggunakan fungsi merge."""

all_film = pd.merge(all_film_name, tags[['movieId','tag']], on='movieId', how='left')
all_film

"""dilakukan pengecekan data kembali , apakah data memiliki nilai null atau tidak"""

all_film.isnull().sum()

"""##<center> Data Preparation </center>
Ternyata data memiliki nilai null pada kolom tag , oleh sebab itu sekarang kita akan mencoba mengatasi missing value pada fitur . 

Berdasarkan pertimbangan kita tidak dapat mengidentifikasi missing value ini berada pada film mana saja , maka langkah terbaik untuk saat ini adalah dengan melakukan drop pada data yang missing . 

Menggunakan fungsi dropna untuk menghapus missing value.
"""

all_film_clean = all_film.dropna()
all_film_clean

"""Memastikan apakah setelah di drop semua missing value telah hilang ."""

all_film_clean.isnull().sum()

"""seperti yang terlihat bahwa sekarang data sudah bersih , pada langkah selanjutnya aku hanya ingin melihat jumlah baris dan kolom pada all_film_clean"""

all_film_clean.shape

"""Selanjutnya adalah mengurutkan judul film untuk melakukan penyamaan data ,agar nantinya tidak terjadi bias pada data . 

disini saya menggunakan fungsi sort.values dan memggunakan movieId sebagai acuannya
"""

# Mengurutkan resto berdasarkan movieId kemudian memasukkannya ke dalam variabel fix_film
fix_film = all_film_clean.sort_values('movieId', ascending=True)
fix_film

"""melakukan perhitungan jumlah data yang sudah fix dan data yang unik ."""

# Mengecek berapa jumlah fix_film
len(fix_film.movieId.unique())

"""ternyata terdapat 1554 id film yang unik dan akan kita gunakan.


Selanjutnya adalah proses preparation oleh sebab itu kita akan membuat definisi preparation dimana isi datanya merupakan isi data dari fix_film .dan data diurutkan berdasarkan movieId
"""

# Membuat variabel preparation yang berisi dataframe fix_film kemudian mengurutkan berdasarkan movieId
preparation = fix_film
preparation.sort_values('movieId')

"""selanjutnya adalah melakukan drop pada data yang berduplikat , karena seperti yang dilihat banyak sekali data yang berduplikat , untuk acuannya kita masih menggunakan movieId."""

# Membuang data duplikat pada variabel preparation
preparation = preparation.drop_duplicates('movieId')
preparation

"""Selanjutnya adalah melakukan konversi data series menjadi list , dengan menggunkan fungsi tolist() dari library numpy"""

# Mengonversi data series ‘movieID’ menjadi dalam bentuk list
film_id = preparation['movieId'].tolist()
 
# Mengonversi data series ‘title’ menjadi dalam bentuk list
film_title = preparation['title'].tolist()
 
# Mengonversi data series ‘tag’ menjadi dalam bentuk list
film_tag = preparation['tag'].tolist()

# Mengonversi data series ‘genre’ menjadi dalam bentuk list
film_genre = preparation['genres'].tolist()
 
print(len(film_id))
print(len(film_title))
print(len(film_tag))
print(len(film_genre))

"""Tahap berikutnya, kita akan membuat dictionary untuk menentukan pasangan key-value pada data film_id, film_title,film_tag dan film_genre yang telah kita siapkan sebelumnya"""

# Membuat dictionary untuk data ‘film_id’, ‘film_title’, dan ‘film_tag’,'film_genre
film_new = pd.DataFrame({
    'id': film_id,
    'title': film_title,
    'tags': film_tag,
    'genres': film_genre
})
film_new

"""##<center> Model Development </center>
- Menggunakan Content Based Filtering .

Pada tahap ini, kita akan membangun sistem rekomendasi sederhana berdasarkan genre yang ada pada film. dengan menggunakan teknik TF-IDF Vectorizer yang berguna untuk menemukan representasi fitur penting pada setiap kategori.

Menggunakan fungsi tfidvectorizer dari library sklearn .
"""

from sklearn.feature_extraction.text import TfidfVectorizer
 
# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
 
# Melakukan perhitungan idf pada data tag
tf.fit(film_new['genres']) 
 
# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names()

"""Melakukan fit dan tranformasi dalam bentuk matriks"""

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(film_new['genres']) 
 
# Melihat ukuran matrix tfidf
tfidf_matrix.shape

"""Perhatikanlah, matriks yang ini miliki berukuran (1554, 24). Nilai 1554 merupakan ukuran data dan 24 merupakan matrik genre. 

selanjutnya untuk menghasilkan vektor tf-idf dalam bentuk matriks, saya menggunakan fungsi todense(). Jalankan kode berikut.
"""

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

"""Selanjutnya adalah berupaya melihat matriks tf-idf untuk beberapa film dan genre ."""

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan jenis genres
# Baris diisi dengan title film
 
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names(),
    index=film_new.title
).sample(22, axis=1).sample(10, axis=0)

"""Output matriks tf-idf di atas menunjukkan bahwa film dengan judul "Upside Down: The Creation Records Story (2010)" memiliki genre Documentary . Hal ini dapat terlihat dari nilai matriks yang menunjukan nilai 1.0 ,selanjutnya ada film "Ikiru (1952)" yang bergenre drama ,film "Cold Comfort Farm (1995)" yang bergenre comedy , dan demikian seterusnya .

Sampai di sini, kita telah berhasil mengidentifikasi representasi fitur penting dari setiap kategori dengan fungsi tfidfvectorizer. Kita juga telah menghasilkan matriks yang menunjukkan korelasi antara genre dengan film.

Selanjutnya, kita akan menghitung derajat kesamaan antara satu judul film dengan film lainnya untuk menghasilkan kandidat film mana yang akan direkomendasikan.

###<center> Cosine Similarity</center>
Sekarang, saya akan menghitung derajat kesamaan (similarity degree) antar film dengan teknik cosine similarity. Di sini, saya menggunakan fungsi cosine_similarity dari library sklearn.
"""

from sklearn.metrics.pairwise import cosine_similarity
 
# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim

"""Pada tahapan ini, proses menghitung cosine similarity dataframe tfidf_matrix yang di peroleh pada tahapan sebelumnya. Dengan satu baris kode untuk memanggil fungsi cosine similarity dari library sklearn, telah berhasil menghitung kesamaan (similarity) antar film. Kode di atas menghasilkan keluaran berupa matriks kesamaan dalam bentuk array. 

Selanjutnya,  melihat matriks kesamaan setiap film dengan menampilkan judul film dalam 5 sampel kolom (axis = 1) dan 10 sampel baris (axis=0). Jalankan kode berikut.
"""

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama resto
cosine_sim_df = pd.DataFrame(cosine_sim, index=film_new['title'], columns=film_new['title'])
print('Shape:', cosine_sim_df.shape)
 
# Melihat similarity matrix pada setiap film
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""Dengan cosine similarity, saya berhasil mengidentifikasi kesamaan antara satu judul film dengan judul film lainnya. Shape (1554, 1554) merupakan ukuran matriks similarity dari data yang di miliki. Berdasarkan data yang ada, matriks di atas sebenarnya berukuran 1554 judul film x 1554 judul film (masing-masing dalam sumbu X dan Y). Artinya, mengidentifikasi tingkat kesamaan pada 1554 judul film. Tapi tentu tidak bisa menampilkan semuanya. Oleh karena itu, saya hanya memilih 10 film pada baris vertikal dan 5 film pada sumbu horizontal seperti pada contoh di atas. 

Angka 1.0 yang berada di tabel mengindikasikan bahwa film pada kolom X (horizontal) memiliki kesamaan dengan film pada baris Y (vertikal). Sebagai contoh, film dengan judul "Searching for Bobby Fischer (1993)" teridentifikasi sama (similar) dengan film "They Shoot Horses, Don't They? (1969)". Contoh lain,judul film Syriana (2005) teridentifikasi mirip dengan judul film " Mean Creek (2004)" dan "All the President's Men (1976)" .

Selanjutnya adalah mencoba mendapatkan rekomendasi .

Di sini, saya membuat fungsi film_recommendations dengan beberapa parameter sebagai berikut:

- title : judul film (index kemiripan dataframe).

- Similarity_data : Dataframe mengenai similarity yang telah saya definisikan sebelumnya.

- Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘title’ dan ‘genre’.

- k : Banyak rekomendasi yang ingin diberikan.

Definisi sistem rekomendasi yang menyatakan bahwa keluaran sistem ini adalah berupa top-N recommendation. Oleh karena itu, saya akan memberikan sejumlah rekomendasi judul film pada pengguna yang diatur dalam parameter k.
"""

def film_recommendations(title, similarity_data=cosine_sim_df, items=film_new[['title', 'genres']], k=5):
    """
    Rekomendasi film berdasarkan kemiripan dataframe
 
    Parameter:
    ---
    title : tipe data string (str)
                judul film (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan film sebagai 
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---
 
 
    Pada index ini, kita mengambil k dengan nilai similarity terbesar 
    pada index matrix yang diberikan (i).
    """
 
 
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,title].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop title agar judul film yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(title, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

"""Selanjutnya, mari kita terapkan kode di atas untuk menemukan rekomendasi judul film yang memiliki genre yang sama dengan film 'Far from Heaven (2002)'

Pertama melakukan pengecekan dulu apakah film 'Far from Heaven (2002)' itu ada pada data atau tidak
"""

film_new[film_new.title.eq('Far from Heaven (2002)')]

"""Berikutnya karena data film sudah ada saatnya melakukan percobaan untuk mendapatkan rekomendasi film . """

# Mendapatkan rekomendasi film yang mirip dengan 
film_recommendations('Far from Heaven (2002)')

"""Dapat dilihat bahwa film 'Far from Heaven (2002)' memiliki genre Drama | Romance dan saat kita mencoba mencari rekomendasi pun yang ditampilkan adalah data film yang memiliki genre yang sama . ini berarti kita telah berhasil membuat sistem rekomendasi dengan model content based filtering .

- Menggunakan model Collaborative Filtering

collaborative filtering membutuhkan data rating dan user . tujuan menggunakan model ini adalah untuk mendapatkan rekomendsi judul film yang sesuai refrensi pengguna berdasarkan rating yang pernah dibuat sebelumnya .

##<center> Data Understanding</center>

Pertama adalah melakukan import library - library yang dibutuhkan
"""

# Import library
import pandas as pd
import numpy as np 
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

"""selanjutnya kita akan melihat data rating kita guna memhami terlebih dahulu data rating yang kita miliki . disini saya mendefinisikan ulang variabel pada data rating agar tidak tertukar dengan fitur rating sebelumnya . disini saya mendefinisikan dengan df ."""

# Membaca dataset
 
df = rating
df

"""##<center> Data Preparation </center>

selanjutnya adalah melakukan persiapan data untuk menyandikan (encoding) fitur userId dan movieId kedalam indeks integer dengan merubahnya menjadi bentuk list terlebih dahulu lalu di encoding

Pertama akan kita lakukan pada data userId terlebih dahulu
"""

# Mengubah userId menjadi list tanpa nilai yang sama
user_ids = df['userId'].unique().tolist()
print('list userId: ', user_ids)
 
# Melakukan encoding userId
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userId : ', user_to_user_encoded)
 
# Melakukan proses encoding angka ke ke userID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userId: ', user_encoded_to_user)

"""Selanjutnya lakyukan hal yang sama pada fitur movieId"""

# Mengubah movieId menjadi list tanpa nilai yang sama
film_ids = df['movieId'].unique().tolist()
 
# Melakukan proses encoding movieId
film_to_film_encoded = {x: i for i, x in enumerate(film_ids)}
 
# Melakukan proses encoding angka ke movieId
film_encoded_to_film = {i: x for i, x in enumerate(film_ids)}
 
#Selanjutnya, petakan userId dan movieId ke dataframe yang berkaitan.
 
# Mapping userId ke dataframe user
df['user'] = df['userId'].map(user_to_user_encoded)
 
# Mapping placeID ke dataframe resto
df['movieId'] = df['movieId'].map(film_to_film_encoded)

"""selanjutnya adalah mencoba menghitung beberapa data , seperti jumlah user , jumlah judul film ,nilai maksimal dan minimal ."""

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print('Jumlah keseluruhan user adalah : ',(num_users))
 
# Mendapatkan jumlah film
num_film = len(film_encoded_to_film)
print('Jumlah keseluruhan film adalah :',(num_film))
 
# Mengubah rating menjadi nilai float
df['rating'] = df['rating'].values.astype(np.float32)
 
# Nilai minimum rating
min_rating = min(df['rating'])
 
# Nilai maksimal rating
max_rating = max(df['rating'])
 
print('Number of User: {}, Number of Film: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_film, min_rating, max_rating
))

"""Selanjutnya adalah melakukan pembagian data untuk training dan validasi.
sebelum itu lakukan proses untuk mengacak data terlebih dahulu agar distribusinya menjadi random .


"""

# Mengacak dataset
df = df.sample(frac=1, random_state=42)
df

"""Selanjutnya, saya akan membagi data train dan validasi dengan komposisi 80:20. sebelumnya itu saya akan memetakan (mapping) data user dan movieId menjadi satu value terlebih dahulu. Lalu, membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training. """

# Membuat variabel x untuk mencocokkan data user dan film menjadi satu value
x = df[['user', 'movieId']].values
 
# Membuat variabel y untuk membuat rating dari hasil 
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
 
print(x, y)

"""Selanjutnya adalah memasuki proses training dimana pada tahap ini model menghitung skor kecocokan antara pengguna dan film dengan teknik embedding.

Pertama, saya akan  melakukan proses embedding terhadap data user dan film. lalu melakukan operasi perkalian dot product antara embedding user dan film. setelah itu menambahkan bias untuk setiap user dan film. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

Di sini juga saya  membuat class RecommenderNet dengan keras Model class. 
"""

class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_users, num_film, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_film = num_film
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.film_embedding = layers.Embedding( # layer embeddings film
        num_film,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.film_bias = layers.Embedding(num_film, 1) # layer embedding resto bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    film_vector = self.film_embedding(inputs[:, 1]) # memanggil layer embedding 3
    film_bias = self.film_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_film = tf.tensordot(user_vector, film_vector, 2) 
 
    x = dot_user_film + user_bias + film_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

"""selanjutnya melakukan proses compfile terhadap model dengan menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. """

model = RecommenderNet(num_users, num_film, 50) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""Memulai proses training , dengan menggunakan batch_size sebesar 32 dan pemanggilan epoch sebanyak 100 kali ."""

# Memulai training
 
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 32,
    epochs = 100,
    validation_data = (x_val, y_val)
)

"""Untuk melihat visualisasi proses training, saya menggunakan metrik evaluasi dengan matplotlib ."""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""Dari proses ini, kita memperoleh informasi bahwa nilai error akhir sebesar sekitar 0.18 dan error pada data validasi sebesar 0.20.


selanjutnya adalah kita akan mencoba mendapatkan rekomendasi film dengan cara mengambil sampel user secara acak dan definisikan variabel film_not_choices yang merupakan daftar film yang belum pernah dilihat oleh pengguna. karena nantinya film - film yang tidak pernah dilihat ini akan direkomendasikan kepada user .

seperti yang terlihat sebelumnya ,pengguna telah memberi rating pada beberapa film yang telah mereka lihat. Kita menggunakan rating ini untuk membuat rekomendasi film yang mungkin cocok untuk pengguna. Nah, film yang akan direkomendasikan tentulah film yang belum pernah dilihat oleh pengguna. Oleh karena itu, kita perlu membuat variabel film_not_choices sebagai daftar film untuk direkomendasikan pada pengguna. 


"""

film_df = film_new
df = pd.read_csv('/content/movie_recommendation/ml-latest-small/ratings.csv')
 
# Mengambil sample user
user_id = df.userId.sample(1).iloc[0]
film_choices_by_user = df[df.userId == user_id]
 
# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html 
film_not_choices = film_df[~film_df['id'].isin(film_choices_by_user.movieId.values)]['id'] 
film_not_choises = list(
    set(film_not_choices)
    .intersection(set(film_to_film_encoded.keys()))
)
 
film_not_choices = [[film_to_film_encoded.get(x)] for x in film_not_choices]
user_encoder = user_to_user_encoded.get(user_id)
user_film_array = np.hstack(
    ([[user_encoder]] * len(film_not_choices), film_not_choices)
)

"""Selanjutnya, adalah mencoba memperoleh rekomendasi film , dengan menggunakan fungsi model.predict() dari library Keras ."""

ratings = model.predict(user_film_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_film_ids = [
    film_encoded_to_film.get(film_not_choices[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Film with high ratings from user')
print('----' * 8)
 
top_film_user = (
    film_choices_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)
 
film_df_rows = film_df[film_df['id'].isin(top_film_user)]
for row in film_df_rows.itertuples():
    print(row.title, ':', row.genres)
 
print('----' * 8)
print('Top 10 film recommendation')
print('----' * 8)
 
recommended_film = film_df[film_df['id'].isin(recommended_film_ids)]
for row in recommended_film.itertuples():
    print(row.title, ':', row.genres)