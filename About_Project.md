# Laporan Proyek Machine Learning - Sarah Salsabila
## Daftar Isi

-   [Project Overview](#project-overview)
-   [Business Understanding](#business-understanding)
-   [Data Understanding](#data-understanding)
-   [Data Preparation](#data-preparation)
-   [Modeling](#modeling)
-   [Evaluation](#evaluation)


## proyek Overview
Pada proyek kali ini, saya akan mencoba membuat sebuah sistem rekomendasi mengenai film .menurut wikipedia film merupakan serangkaian gambar diam, yang ketika ditampilkan pada layar akan menciptakan ilusi gambar bergerak karena efek fenomena phi. Ilusi optik ini memaksa penonton untuk melihat gerakan berkelanjutan antar objek yang berbeda secara cepat dan berturut-turut.

![Menonton Film](https://ds393qgzrxwzn.cloudfront.net/resize/m720x480/cat1/img/images/0/SK2kCfuPtH.jpg)

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

    Kelebihan:
    
    - Teknik ini baik dipakai ketika skala user yang besar.
    - Teknik ini dapat menemukan ketertarikan spesifik dari seorang user, dan dapat -  merekomendasikan item yang jarang disukai orang lain.
    
    Kekurangan:
    - Karena meta feature yang digunakan menentukan sendiri, kualitas dari rekomendasi tergantung kualitas dari meta feature itu sendiri.

- **Collaborative Filtering**
 Teknik ini merekomendasikan item yang mirip dengan preferensi pengguna pada masa lalu. Teknik ini membutuhkan data rating dari user untuk menghasilkan rekomendasi sejumlah film yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna, kita akan mengidentifikasi film -film yang mirip dan belum pernah dilihat oleh pengguna untuk direkomendasikan. 

## Data Understanding
Pada proyek machine learning ini saya menggunakan dataset mengenai daftar film untuk sistem rekomendasi . yaitu Movie Recommendation Data yang dapat diakses pada link berikut
[movie recommendation data](https://www.kaggle.com/rohan4050/movie-recommendation-data)
Setelah di unzip terlihat bahwa didalam file movie-recommendation data terdapat 5 buah berkas csv . diantaranya :
- links.csv
- movies.csv
- ratings.csv
- tags.csv
- movies_metadata.csv

Pada proyek machine learning kali ini saya hanya menggunakan 3 file csv saja , yaitu tags,rating dan movies .

Variabel-variabel pada Movie-recommendation dataset yang digubnakan adalah sebagai berikut:
- movieId   : merupakan id dari judul film tertentu.
- userId    : merupakan id dari pengguna yang melakukan penilaian dan memberikan tag pada suatu film.
- tag       : tag yang diberikan pengguna terkait film tersebut.
- genres    : merupakan genre film.
- title     : merupakan judul film.

### Data Loading
Untuk mempermudah dalam proses memahami data , saya melakukan proses loading data terlebih dahulu .
- Melakukan install kaggle dengan library pip .
    ![PipKaggle](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/1.%20pip.jpg)
- Selanjutnya melakukan impor file pada google colab dengan menginputkan data API akun kaggle saya .
- Membuat direktori baru untuk menyimpan install data kaggle.json.
    ![import](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/2.%20kagglejson.jpg)
- Mengunduh dataset yang akan digunakan.
    ![Download dataset](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/3.download.jpg)
- Setelah mengunduh data ,langkah berikutnya adalah melakukan unzip pada data . di sini saya menyimpan data yang sudah di unzip pada berkas d dengan nama folder movie - recommendation .
    ![unzip](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/4.%20unzip.jpg)
- Lakukan impor library pandas , karena kita akan menggunakan fungsi read_csv yang terdapat didalam library pandas . untuk file movies.csv saya definisikan sebagai dataframe movie , lalu tags.csv sebagai dataframe tag dan untuk ratings.csv sebagai dataframe rating.
    ![readcsv](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/5.%20read_csv.jpg)

    Setelah itu saya melakukan pemahaman lebih mengenai data dengan mencari tahu jumlah baris dan kolom , lalu jumlah data tiap variabel , mengecek apakah terdapat nilai null dan nilai berduplikat.
- Memanggil fungsi columns untuk melihat apa saja nama tabel pada setiap data .
    ![kolom](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/6.%20Cek%20kolom.jpg)
    Informasi yang dapat diambil dari hasil ekesekusi kode tersebut adalah 
    - Dataframe movie memiliki 3 buah kolom yang terdiri dari ['movieId', 'title', 'genres'] .
    - Dataframe tags memiliki 4 buah kolom terdiri dari ['userId', 'movieId', 'tag', 'timestamp'].
    - Dataframe rating memiliki 4 buah kolom juga yaitu ['userId', 'movieId', 'rating', 'timestamp'].

- Melakukan pengecekan jumlah data, baris dan kolom pada setiap file .
    ![lendata](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/7.%20jumlah%20data.jpg)

    Seperti yang terlihat bahwasannya dataframe movie memiliki 9742 baris dan 3 kolom , tags memiliki 3683 baris dan 4 kolom , dan rating dengan 100836 baris dengan 4 kolom .
- langkah berikutnya adalah pengecekan apakah data memiliki nilai null , duplicat dan mencari tahu informasi lebih dalam tentang data melalui fungsi describe atau info .

   *1. Data Movie*
    ![datamovie](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/8.%20movie.ndi.jpg)
    Dari hasil eksekusi didapatkan informasi sebagai berikut :
    Pada dataframe movie 
    - Tidak terdapat nilai yang berduplikat.
    - Tidak memiliki nilai null.
    - Movie memiliki 2 tipe data yaitu 2 bertipe data object dan 1 bertipe data int
    - Movie memiliki berbagai macam kombinasi genre.
    
     *2. Data Tags*
    ![datatags](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/9.tag.ndi.jpg)
    ![datatagsdrop](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/10.%20tah%20drop.jpg)
    Dari hasil eksekusi kode - kode di atas , didapatkan informasi sebagai berikut :
    Pada dataframe tags
    - Tidak terdapat nilai yang berduplikat.
    - Tidak memiliki nilai null.
    - tags memiliki 2 tipe data yaitu 1 bertipe data object dan 3 bertipe data int.
    - tags sekarang memiliki 3 kolom karena kolom timestamp telah dihapus .
    
    *3.Data Rating*
    ![datarating](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/11.%20rating%20head.jpg)  
    ![datarating](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/12.%20rating%20desc.jpg)
    ![datararingdrop](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/13%20.%20drop%20rating.jpg)
    Dari hasil eksekusi kode - kode di atas , didapatkan informasi sebagai berikut :
    Pada dataframe tags
    - Dataframe rating sekarang memiliki 3 kolom , karena kolom timestamp dihapus.
    - terdapat 610 jumlah data pada kolom userId.
    - tedapat 9724 jumlah data pada kolom movieId.
    - terdapat 100836 jumlah data pada rating .
    
### Data preprocessing
Pada tahap ini yang pertama saya lakukan adalah melakukan penggabungan dataframe dengan menggunakan fungsi concatenate dari library numpy . alasannya adalah variabel yang kita butuhkan untuk melakukan analisis itu terdapat pada dataframe yang berbeda , oleh sebab itu kita harus menggabungkan dataframe dengan menggunakan data yang unik agar saat digabungkan data saling sinkron antara data pada dataframe 1 dan dataframe yang lainnya . 
- Pertama melakukan concatenate pada dataframe movie dan tags berdasarkan variabel atau fitur movieId yang unik .
![concatenatemovieId](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/14%20.%20preparation%20nump%20impor.jpg)
    Dari hasil eksekusi di dapatkan informasi bahwa file pada kategori film ketika digabung memiliki 9742 file movie yang unik.

- Selanjutnya melakukan penggabungan dataframe berdasarkan variabel atau fitur userId sebagai acuan atau kunci dalam penggabungan .Dataframe yang digabung adalah rating dan tags . sama sebelumnya dengan mengambil nilai data yang unik .
![concatenateuserId](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/15.%20concatenate%20user.jpg)
     Seperti yang terlihat pada hasil eksekusi sebelumnya bahwa hanya terdapat 610 data pengguna dari 8742 film yang memiliki rating .

- Setelah menggabungkan dataframe berdasarkan variabel yang kita butuhkan yaitu userId dan movieId .Berikutnya adalah upaya untuk mengetahui jumlah rating yang diberikan oleh pengguna , untuk mengetahuinya kita akan melakukan pendefinisian baru tentang ini dengan nama movieinfo . movieinfo akan berisikan kolom - kolom yang kita perlukan seperti title,genre,tags,userId dan movieId ,di mana untuk mendapatkan kolom - kolom tersebut kita melakukan concat ketiga dataframe .lalu melakukan merge dataframe rating dan movieinfo yang sudah dilakukan concat berdasarkan movieId .
    ![concat&merge](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/16.%20concat.jpg)
- Setelah itu dilakukan pengecekan pada dataframe yang telah digabung (film) apakah data memiliki nilai null,
    ![null](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/17.%20filmnull1.jpg)

    Dari kedua proses eksekusi di atas didapatkan informasi bahwa 
    -   Terdapat nilai null pada beberapa kolom , kolom yang bersih dari nilai null adalah userId_x , movieId, dan rating_x.
- Selanjutnya menggabungkan dataframe file dengan menggunakan fungsi grupby berdasarkan variabel movieId
    ![groupby](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/18.grupby.jpg)
- Selanjutnya adalah membuat definisi all_film_rate dengan variabel rating yang sudah diketahui sebelumnya 
    ![defallfilmrate](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/19.%20allfilmrate.jpg)
- Setelah mengabungkan fitur mengenai film dan genre serta ratingnya selanjutnya adalah menggabungnya dengan dataframe tags untuk mendapatkan userId dan tag label apa saja yang diberikan user pada film - film tersebut .masih dengan menggunakan fungsi merge.
    ![meregeallfilmrate](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/20.%20merge%20allfilmrate.jpg)
    ![meregeallfilmratetag](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/21.%20merge%20tah%20allfilmrate.jpg)
- Dilakukan pengecekan data kembali , apakah data memiliki nilai null atau tidak
    ![isnuulallfilmrate](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/22.%20allfim%20isnull.jpg)
selanjutnya saya akan melakukan pembersihan missing value. Pada penjelasan langkah sebelumnya di dapatkan informasi bahwa data tag memiliki missing value . Berdasarkan pertimbangan kita tidak dapat mengidentifikasi missing value ini berada pada film mana saja , maka langkah terbaik untuk saat ini adalah dengan melakukan drop pada data yang missing . 
- Menggunakan fungsi dropna untuk menghapus missing value.
    ![dropnull](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/23.dropna%20isnull.jpg)
- Memastikan apakah setelah di drop semua missing value telah hilang .
    ![clean](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/24.%20cleannull.jpg)
- Langkah selanjutnya saya hanya ingin melihat jumlah baris dan kolom pada all_film_clean
    ![jumlahfilmclean](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/25.%20sort%20values%20by%20movie.jpg)
- Mengurutkan judul film untuk melakukan penyamaan data ,agar nantinya tidak terjadi bias pada data .
    ![sortvalue](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/25.%20sort%20values%20by%20movie.jpg)
- melakukan perhitungan jumlah data yang sudah fix dan data yang unik .
 ternyata terdapat 1554 id film yang unik dan akan kita gunakan.
    ![len data](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/26.%20len%20film.jpg)
##Data Preparation
- Selanjutnya adalah memasuki babak preparation di mana di sini saya membuat definisi preparation. di mana isi datanya merupakan isi data dari fix_film .dan data diurutkan berdasarkan movieId.
    ![preparation](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/27.%20preparation%20dev.jpg)
- Melakukan drop pada data yang berduplikat , karena seperti yang dilihat banyak sekali data yang berduplikat , untuk acuannya kita masih menggunakan movieId.
    ![dropduplikat](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/28.%20prepa%20drop.jpg)
- Setelah memastikan data sudah bersih selanjutnya adalah proses mengkonversi data series menjadi list . dengan menggunakan fungsi tolist() dari library numpy .
    ![ubahlist](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/29.%20ubah%20list%20.jpg)
    Dapat dilihat bahwa jumlah data pada setiap variabel dalam bentuk list sudah sama yaitu 1554 data .
- Membuat dictionary untuk menentukan pasangan key-value pada data film_id, film_title,film_tag dan film_genre yang telah kita siapkan sebelumnya.
    ![dictionary](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/30%20.%20directory%20film_new.jpg)

## Modeling
Pada tahapan ini saya menggunakan 2 model untuk dapat memenuhi goals saya pada saat bussines understanding .
1. Menggunakan model content based filtering
2. Menggunakan collaborative filtering

- Content Based Filtering

#### Model

 Pada tahap ini, kita akan membangun sistem rekomendasi sederhana berdasarkan genre yang ada pada film. dengan menggunakan teknik TF-IDF Vectorizer yang berguna untuk menemukan representasi fitur penting pada setiap kategori.
  - Menggunakan fungsi tfidvectorizer dari library sklearn .
      ![tfidvector](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/31.tfidvectorizer.jpg)
 - Melakukan fit dan tranformasi dalam bentuk matriks
     ![matriks](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/32.%20matriks.jpg)
    Perhatikanlah, matriks yang ini miliki berukuran (1554, 24). Nilai 1554 merupakan ukuran data dan 24 merupakan matriks genre.
 - menghasilkan vektor tf-idf dalam bentuk matriks, saya menggunakan fungsi todense()
     ![todense](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/33.todense.jpg)
 - Selanjutnya adalah berupaya melihat matriks tf-idf untuk beberapa film dan genre .
     ![tf dataframe](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/34.%20replace.jpg)
    Output matriks tf-idf di atas menunjukkan bahwa film dengan judul "Upside Down: The Creation Records Story (2010)" memiliki genre Documentary . Hal ini dapat terlihat dari nilai matriks yang menunjukkan nilai 1.0 ,selanjutnya ada film "Ikiru (1952)" yang bergenre drama ,film "Cold Comfort Farm (1995)" yang bergenre comedy , dan demikian seterusnya .

Sampai di sini, artinya telah berhasil mengidentifikasi representasi fitur penting dari setiap kategori dengan fungsi tfidfvectorizer. Juga telah menghasilkan matriks yang menunjukkan korelasi antara genre dengan film.

Selanjutnya, akan menghitung derajat kesamaan antara satu judul film dengan film lainnya untuk menghasilkan kandidat film mana yang akan direkomendasikan. 
-  Menghitung derajat kesamaan (similarity degree) antar film dengan teknik cosine similarity. Di sini, saya menggunakan fungsi cosine_similarity dari library sklearn.
    ![cosine](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/35.cosine%20.jpg)
 Pada tahapan ini, proses menghitung cosine similarity dataframe tfidf_matrix yang di peroleh pada tahapan sebelumnya. Dengan satu baris kode untuk memanggil fungsi cosine similarity dari library sklearn, telah berhasil menghitung kesamaan (similarity) antar film. Kode di atas menghasilkan keluaran berupa matriks kesamaan dalam bentuk array. 
- Selanjutnya,  melihat matriks kesamaan setiap film dengan menampilkan judul film dalam 5 sampel kolom (axis = 1) dan 10 sampel baris (axis=0).
    ![df cosine](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/36.%20df%20cosine.jpg)

Dengan cosine similarity, saya berhasil mengidentifikasi kesamaan antara satu judul film dengan judul film lainnya. Shape (1554, 1554) merupakan ukuran matriks similarity dari data yang di miliki. Berdasarkan data yang ada, matriks di atas sebenarnya berukuran 1554 judul film x 1554 judul film (masing-masing dalam sumbu X dan Y). Artinya, mengidentifikasi tingkat kesamaan pada 1554 judul film. tetapi tentu tidak bisa menampilkan semuanya. Oleh karena itu, saya hanya memilih 10 film pada baris vertikal dan 5 film pada sumbu horizontal seperti pada contoh di atas. 

Angka 1.0 yang berada di tabel mengindikasikan bahwa film pada kolom X (horizontal) memiliki kesamaan dengan film pada baris Y (vertikal). Sebagai contoh, film dengan judul "Searching for Bobby Fischer (1993)" teridentifikasi sama (similar) dengan film "They Shoot Horses, Don't They? (1969)". Contoh lain,judul film Syriana (2005) teridentifikasi mirip dengan judul film " Mean Creek (2004)" dan "All the presiden's Men (1976)" . 

#### result
Pada tahap ini kita kan mencoba mendapatkan rekomendasi .Di sini, saya membuat fungsi film_recommendations dengan beberapa parameter sebagai berikut:

    title : judul film (index kemiripan dataframe).
    
    Similarity_data : Dataframe mengenai similarity yang telah saya definisikan sebelumnya.
    
    Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘title’ dan ‘genre’.
    
    k : Banyak rekomendasi yang ingin diberikan.

Definisi sistem rekomendasi yang menyatakan bahwa keluaran sistem ini adalah berupa top-N recommendation. Oleh karena itu, saya akan memberikan sejumlah rekomendasi judul film pada pengguna yang diatur dalam parameter k. 
![def](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/37.%20def%20recom.jpg)

Selanjutnya untuk mendapatkan rekomendasi judul film yang memiliki genre yang sama dengan suatu film , di sini saya akan menggunakan film 'Far from Heaven (2002)'.
jalankan kode berikut agar tahu apakah film ini ada pada dataset .
![def](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/38.%20cek%20film%20adaga.jpg)
Berikutnya karena data film terdeteksi , dan terlihat bahwa film 'Far from Heaven (2002)' merupakan film dengan genre drama|romance. saatnya melakukan percobaan untuk mendapatkan rekomendasi film .
![result](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/39.%20content%20based%20result.jpg)
Dapat dilihat bahwa film 'Far from Heaven (2002)' memiliki genre Drama | Romance dan saat mencoba mencari rekomendasi pun yang ditampilkan adalah data film yang memiliki genre yang sama . ini menandakan bahwa telah berhasil membuat sistem rekomendasi dengan model content based filtering .

- collaborative filtering 

#### Model

Pada model ini saya akan menggunakan data rating dan user . tujuan menggunakan model ini adalah untuk mendapatkan rekomendsi judul film yang sesuai refrensi pengguna berdasarkan rating yang pernah dibuat sebelumnya .
- Hal pertama adalah impor library yang dibutuhkan 
- selanjutnya ,memasuki tahap data understanding. saya akan melihat data rating guna memhami terlebih dahulu data rating yang di miliki . di sini saya mendefinisikan ulang variabel pada data rating agar tidak tertukar dengan fitur rating sebelumnya . di sini saya mendefinisikan dengan df .
    ![impordandf](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/40.collaborative%201.jpg)
- selanjutnya memasuki tahap data preparation , saya akan melakukan persiapan data untuk menyandikan (encoding) fitur userId dan movieId kedalam indeks integer dengan merubahnya menjadi bentuk list terlebih dahulu lalu di encoding.

    ![encoding](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/41.%20coll%202.jpg)
- setelah melakukan encoding pada variabel userId dan movieId , selanjutnya saya mencoba menghitung jumlah film,user,rating,tags,minimal rating dan maximal rating .
    ![len](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/42.%20jumlah%20user.jpg)
 Seperti yang terlihat , jumlah keseluruhan user adalah 610 data , keseluruhan film adalah 9724 data ,minimal rating adalah 0.5 dan maximal rating adalah 5.
- Langkah selanjutnya adalah melakukan pembagian data untuk training dan validasi. sebelum itu lakukan proses untuk mengacak data terlebih dahulu agar distribusinya menjadi random .
    ![acak data](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/43.%20coll%203%20acalk%20dataset.jpg)
- Saya akan membagi data train dan validasi dengan komposisi 80:20. sebelumnya itu saya akan memetakan (mapping) data user dan movieId menjadi satu value terlebih dahulu. Lalu, membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training.
    ![bagi data](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/44.%20bagi%20data%20col4.jpg)
- Selanjutnya adalah memasuki proses training di mana pada tahap ini model menghitung skor kecocokan antara pengguna dan film dengan teknik embedding.
    Pertama, saya akan melakukan proses embedding terhadap data user dan film. lalu melakukan operasi perkalian dot product antara embedding user dan film. setelah itu menambahkan bias untuk setiap user dan film. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.
    Di sini juga saya membuat class RecommenderNet dengan keras Model class.
    ![def recom](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/45.%20keras%20model%20recomm%20col%205.jpg)
- selanjutnya melakukan proses compfile terhadap model dengan menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation.
    ![model](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/46.%20model.jpg)
- Memulai proses training , dengan menggunakan batch_size sebesar 32 dan pemanggilan epoch sebanyak 100 kali .
    ![training](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/47.training.jpg)
- Untuk melihat visualisasi proses training, saya menggunakan metrik evaluasi dengan matplotlib .
    ![matplotlib](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/48.%20history.jpg)
     Dari proses ini, kita memperoleh informasi bahwa nilai error akhir sebesar sekitar 0.18 dan error pada data validasi sebesar 0.20.

#### Result 

Selanjutnya adalah kita akan mencoba mendapatkan rekomendasi film dengan cara mengambil sampel user secara acak dan definisikan variabel film_not_choices yang merupakan daftar film yang belum pernah dilihat oleh pengguna. karena nantinya film - film yang tidak pernah dilihat ini akan direkomendasikan kepada user .

Seperti yang terlihat sebelumnya ,pengguna telah memberi rating pada beberapa film yang telah mereka lihat. Kita menggunakan rating ini untuk membuat rekomendasi film yang mungkin cocok untuk pengguna. Nah, film yang akan direkomendasikan tentulah film yang belum pernah dilihat oleh pengguna. Oleh karena itu, kita perlu membuat variabel film_not_choices sebagai daftar film untuk direkomendasikan pada pengguna. ![df](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/49.%20film_df.jpg)
Selanjutnya, adalah mencoba memperoleh rekomendasi film , dengan menggunakan fungsi model.predict() dari library Keras .
![modelpredict](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/50.%20colla%20re.jpg)
![result](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/collaborative%20result.jpg)

Yeay berhasil melakukan sistem rekomendasi film dengan collaborative learning .

## Evaluation
Untuk tahap evaluasi mengenai sistem rekomendasi film dengan menggunakan 2 model machine learning .
1. content based filtering .
Pada content based filtering saya sendiri sudah merasa puas dengan hasil output yang diberikan , karena menurut saya sistem telah dapat memberikan rekomendasi berdasarkan genre yang saya suka sebelumnya . 
untuk mengetahui akurasi dari model , saya membuat nya dengan system precission yaitu :

```
     Akurasi = jumlah item yg direkomendasikan yg memiliki genre yg sama dengan item yg menjadi pusat rekomendasi / jumlah item yg direkomendasikan 
```
atau
```
     Precision = #of recommendation that are relevant/#of item we recommend.
```
Seperti pada proyek saya , ketika saya melakukan pengecekan data untuk mengetahui genre apa film yang saya tonton sebelumnya .

![film](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/38.%20cek%20film%20adaga.jpg)

Di dapatkan informasi bahwa film 'Far from Heaven (2002)' memiliki genre Drama & Romance .

Lalu saat mencari rekomendasi berdasarkan film 'Far from Heaven (2002)', didapatkan hasil seperti berikut :
![result](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/39.%20content%20based%20result.jpg)

Dimana sistem memberikan 5 rekomendasi film lainnya dengan genre yang sama persis dengan genre film sebelumnya yaitu Drama & Romance 

maka Akurasinya adalah ( 5 / 5 = 1 ) . nilai akurasi adalah 1 atau 100%. dimana ini berarti tingkat akurasi dengan content based learning sudah sangat tinggi .

2. Collaborative Filtering

Pada model ini saya menggunakan evaluasi metrik RMSE atau Root Mean Squared Error . dimana RMSE ini adalah sebuah metode pengukuran dengan mengukur perbedaan nilai prediksi sebuah model sebagai estimasi nilai yang di observasi .

Pada RMSE semakin kecil nilai error artinya prediksi semakin akurat atau bagus . dan semakin besar nilai error maka prediksi kurang bagus  atau kurang akurat .

Pada proyek ini operasi penggunaan rmse pada model collaborative filtering terlihat saat membuat kelas recommendedNet dengan library keras . seperti berikut :
![kelasrecommended](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/45.%20keras%20model%20recomm%20col%205.jpg)

lalu dilakukan proses compile model dengan menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer.
dan di lakukan training dengan epoch = 100 dan batch_size = 32.
![training](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/eval1.jpg)
dan didapatkan hasil akhir training seperti berikut :
![last epoch](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/eval%202.jpg)

Hasil visualisasi metriks rmse :
![hist](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/eval3.jpg)

Dapat dilihat bahwa rmse menghasilkan rmse sebesar 0.1 dan val_rmse 0.2 . dimana angka ini merupakan angka yang bagus untuk sistem rekomendasi . seperti penjelasan sebelumnya semakin kecil nilai error rmse maka hasil prediksi semakin baik . 

Dan inilah hasil akhir rekomendasi berdasarkan sistem collaborative filtering .
![result](https://raw.githubusercontent.com/sarahsalsabila01/ImageS2-MLT/main/collaborative%20result.jpg)


Sekian Laporan dari saya mengenai proyek sistem rekomendasi Machine Learning. 
untuk image dapat diakses pada link berikut :
[Github](https://github.com/sarahsalsabila01/ImageS2-MLT)


    
    



