
###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering )  müşteriler kümelere ayrılıp ve davranışları gözlemlenmek istenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# 20.000 gözlem, 13 değişken

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type : 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.


###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv.csv verisini okuyunuz.
           # 2. Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz. Tenure(Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# GÖREV 2: K-Means ile Müşteri Segmentasyonu
           # 1. Değişkenleri standartlaştırınız.
           # 2. Optimum küme sayısını belirleyiniz.
           # 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
           # 4. Herbir segmenti istatistiksel olarak inceleyeniz.

# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
           # 1. Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
           # 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
           # 3. Herbir segmenti istatistiksel olarak inceleyeniz.


###############################################################
# GÖREV 1: Veri setini okutunuz ve müşterileri segmentlerken kullanıcağınız değişkenleri seçiniz.
###############################################################
import itertools

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


df_ = pd.read_csv(r"C:\Users\furka\Desktop\VBO DSMLBC-6\DSMLBC-8\Hafta 9\flo_data_20k.csv")
df = df_.copy()

# tarih değişkenine çevirme
df.head()
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]') # en son kaç gün önce alışveriş yaptı
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
model_df.head()

###############################################################
# GÖREV 2: K-Means ile Müşteri Segmentasyonu
###############################################################

# 1. Değişkenleri standartlaştırınız.
#SKEWNESS
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()

# Normal dağılımın sağlanması için Log transformation uygulanması
model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()


# Scaling (Tercihen)
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()
#model_df.describe()

# 2. Optimum küme sayısını belirleyiniz.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20),timings=False)
elbow.fit(model_df)
#plt.xlim(1,3)
#plt.ylim(0.2,0.4)
elbow.show()
elbow.k_values_
elbow.k_timers_

def timesearcher(start,end,timings=False):
    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(start,end), timings=True)
    elbow.fit(model_df)
    elbow.show()
    k_times = elbow.k_timers_
    k_values = elbow.k_values_
    for i, j in zip(k_times, k_values):
        print(f"{i} saniyede {j} cluster fit edilmiştir.---{i/(j-1)}")
    return k_times,k_values,start,end,elbow


k_times,k_values,start,end,elbow = timesearcher(2,20)

elbow.elbow_value_

# 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments=k_means.labels_
segments
segments[0:5]

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head(10)
final_df["segment"].value_counts()
final_df.shape


# 4. Herbir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","median","count"],
                                  "order_num_total_ever_offline":["mean","median","count"],
                                  "customer_value_total_ever_offline":["mean","median","count"],
                                  "customer_value_total_ever_online":["mean","median","count"],
                                  "recency":["mean","median","count"],
                                  "tenure":["mean","median","count"]})


#Ekstra olarak buradan elde ettiğim faydalı bilgiyi müşteriler ile iletişme geçen ya da kampanya hazırlayan departmana iletmek istersem;
segment_6_musterilerim = final_df[final_df["segment"]== 6]
departmana_gidecek = pd.merge(segment_6_musterilerim,df,how='inner')
departmana_gidecek.to_csv("departmanagidecek.csv", index=False)
departmana_gidecek.shape

"""
Segment-0:  Bu segmentte 5448 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.42;ortanca değeri 1 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.28;ortanca değeri 1 olarak hesaplanmıştır.
Bu segment için online ve offline platformlarda yapılan alışveriş sayıları arasında ortalama değerlere baktığımızda bariz bir fark görmemekle beraber ortanca değerlerinin aynı olduğunu gözlemliyoruz.
>> Müşterilerin online platformlarda yaptığı alışveriş sayısı online platformlarda yaptığı alışveriş sayısında çok az bir farkla da olsa fazladır diyebiliriz.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 145;ortanca değeri 120 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 210;ortanca değeri 179 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  para bırakma potansiyeli hem ortalama hem de median değerlerinde  önemli sayılabilecek bir fark gözlenmektedir.

>> Müşterilerin online platformlarda  şirkete para bırakma potansiyeli  offline platformlardakinden daha fazladır.
>>Tüm segmentler arasında en az alışveriş yapan ve Flo'ya en az para bırakan müşteriler bu segmenttedir.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde flo'dan çok uzun zamandır alışveriş yapmayan müşteriler olduğunu görüyoruz.
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(489) ve median(393)'larını incelediğimizde flo için yeni müşteriler olmadığını görüyoruz. Fakat çok da eski sayılmazlar. :)

Özet: Online plaftormlarda ödedikleri para offline'a göre daha fazla. Belli bir süre alışveriş yapmış fakat uzun zamandır uğramıyor. Flo  için ne çok yeni ne de çok eski müşterilerden oluşuyor.

Aksiyon: Sms ya da e-mail yoluyla bu müşterilerle iletişime geçilebilir. Popüler ürünlerle ilgili indirimler tanımlanabilir.
"""

"""
Segment-1:  Bu segmentte 1841 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 2.20;ortanca değeri 2 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 2.10;ortanca değeri 2 olarak hesaplanmıştır.
Bu segment için online ve offline platformlarda yapılan alışveriş sayıları arasında ortalama değerlere baktığımızda bariz bir fark görmemekle beraber ortanca değerlerinin aynı olduğunu gözlemliyoruz.
>> Müşterilerin online platformlarda yaptığı alışveriş sayısı online platformlarda yaptığı alışveriş sayısında çok az bir farkla da olsa fazladır diyebiliriz.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 293;ortanca değeri 209 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 377;ortanca değeri 297 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  para bırakma potansiyeli  hem ortalama hem de median değerlerinde önemli  bir fark gözlenmektedir.
>> Müşterilerin online platformlarda para bırakma potansiyeli offline platformlardakinden daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde flo'dan en sık alışveriş yapan müşterilerin bu segmentte olduğunu görüyoruz.
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(665) ve median(600)'larını incelediğimizde bu segment flo için en yeni müşterilerin bulunduğu ikinci segmenttir.

Özet: Online plaftormlarda ödedikleri para offline'a göre daha fazla. Flo'dan sıklıkla alışveriş yapıyorlar. Flo için en yeni müşterilerin bulunduğu ikinci segmenttir.

Aksiyon:
"""

"""
Segment-2:  Bu segmentte 3269 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.84;ortanca değeri 1 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 3.77;ortanca değeri 3 olarak hesaplanmıştır.
Bu segment için online ve offline platformlarda yapılan alışveriş sayıları arasında hem ortalama hem de median değerlerinde bariz bir fark gözlemliyoruz.
>> Müşterilerin offline platformlarda yaptığı alışveriş sayısı online platformlarda yaptığı alışveriş sayısından çok daha fazladır.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 551;ortanca değeri 449 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 283;ortanca değeri 222 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  ödediği toplam ücret değişkeni için hem ortalama hem de median değerlerinde önemli  bir fark gözlenmektedir.
>> Müşterilerin offline platformlarda para bırakma potansiyeli online platformlardakinden çok daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en az olan 3. segment. (1,7,3)
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(729) ve median(578)'larını incelediğimizde bu segment flo için en eski müşterilerin bulunduğu 3. segmenttir. (6,7,3)

Özet: Offline platformlarda alışveriş yapma ve şirkete para bırakma potansiyeli en fazla olan segmenttir. Alışveriş yapma sıklığı en az olan 3. segment. (Arttırılmalıdır. Potansiyel fazla. ) Flo için en eski müşterilerin bulunduğu 3. segmenttir.

Aksiyon: geçmiş alışverişlerine dayalı olarak ürünler önerilebilir ve hesaplarına  indirim kuponu tanımlanabilir.
"""

"""
Segment-3:  Bu segmentte 828 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.40;ortanca değeri 1 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.13;ortanca değeri 1 olarak hesaplanmıştır.
Bu segment için online ve offline platformlarda yapılan alışveriş sayıları arasında ortalama değerlere baktığımızda bariz bir fark görmemekle beraber ortanca değerlerinin aynı olduğunu gözlemliyoruz.
>> Müşterilerin offline platformlarda yaptığı alışveriş sayısı online platformlarda yaptığı alışveriş sayısından çok daha fazladır.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 192;ortanca değeri 149 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 250;ortanca değeri 179 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  ödediği toplam ücret değişkeni  için hem ortalama hem de median değerlerinde önemli  bir fark gözlenmektedir.
>> Müşterilerin online platformlarda para bırakma potansiyeli offline platformlardakinden daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en az olan 4. segment. (1,7,3,4)
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(37) ve median(32)'larını incelediğimizde bu segment flo için en yeni müşterilerin bulunduğu segmenttir.

Özet: Online platformlarda alışveriş yapma ve şirkete para bırakma potansiyeli offline platformlara göre daha yüksektir. Alışveriş yapma sıklığı en az olan 4. segment. En yeni müşteriler bu segmenttedir.

Aksiyon: Onlarla sms email vb kanallarla iletişim kurulmaya başlanabilir. bu müşteriler için bir hoşgeldin kampanyası tasarlanabilir.
"""

"""
Segment-4:  Bu segmentte 3340 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.61;ortanca değeri 1 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.48;ortanca değeri 1 olarak hesaplanmıştır.
Bu segment için online ve offline platformlarda yapılan alışveriş sayıları arasında ortalama değerlere baktığımızda bariz bir fark görmemekle beraber ortanca değerlerinin aynı olduğunu gözlemliyoruz.
>> Müşterilerin offline platformlarda yaptığı alışveriş sayısı online platformlarda yaptığı alışveriş sayısından çok daha fazladır.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 185;ortanca değeri 149 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 250;ortanca değeri 206 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  ödediği toplam ücret değişkeni  için hem ortalama hem de median değerlerinde önemli bir fark gözlenmektedir.
>> Müşterilerin online platformlarda ödediği toplam ücret offline platformlarda ödediği toplam ücretten daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en iyi olan 3. segment. (2,6,5)
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(604) ve median(546)'larını incelediğimizde bu segment flo için en eski müşterilerin bulunduğu 5. segmenttir. (6,7,3,2,5)

Özet: Online platformlarda alışveriş yapma ve şirkete para bırakma potansiyeli offline platformlara göre daha yüksektir. Alışveriş yapma sıklığı en iyi olan 3. segment olmasına rağmen en az para bırakan da  3. segmenttir.  En eski müşterilerin bulunduğu 5. segmenttir.

Aksiyon: Sms ya da email ile iletişime geçilip hesaplarına sınırlı süreli indirim kuponları tanımlanabilir.
"""

"""
Segment-5:  Bu segmentte 1974 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 8.63;ortanca değeri 6 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 2.10;ortanca değeri 2 olarak hesaplanmıştır.
Bu segment için online ve offline platformlarda yapılan alışveriş sayıları arasında ortalama ve median değerlerine baktığımızda bariz bir fark gözlemliyoruz.
>> Müşterilerin online platformlarda yaptığı alışveriş sayısı offline platformlarda yaptığı alışveriş sayısından çok daha fazladır.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 278;ortanca değeri 200 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 1475;ortanca değeri 1065 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  ödediği toplam ücret değişkeni  için hem ortalama hem de median değerlerinde çok önemli bir fark gözlenmektedir.
>> Müşterilerin online platformlarda şirkete bıraktığı para offline platformlardakinden çok daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en iyi olan 2. segment. (2,6)
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(966) ve median(752)'larını incelediğimizde bu segment flo için en eski müşterilerin bulunduğu ilk segmenttir.

Özet: Online platformlarda alışveriş yapma ve şirkete para bırakma potansiyeli offline platformlara göre çok daha yüksektir. Alışveriş yapma sıklığı en iyi olan 2. segmenttir. Sık alışveriş yapıyorlar ve yüklü miktarda şirkete para bırakıyorlar.En eski müşterilerin bulunduğu ilk segmenttir. Best customers :)

Aksiyon: Bu müşteriler ödüllendirilebilir. Örneğin yeni çıkan bir ürünün tanıtımını ilk bu müşterilerin görmesi sağlanılabilir. Böylece bu müşteriler ürünleri diğer müşterilere de tanıtıp o ürünü popüler hale getirebilierler.
"""

"""
Segment-6:  Bu segmentte 3245 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 6.37;ortanca değeri 5 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.54;ortanca değeri 1 olarak hesaplanmıştır.
Bu segment için online ve offline platformlarda yapılan alışveriş sayıları arasında ortalama ve median değerlerine baktığımızda bariz bir fark gözlemliyoruz.
>> Müşterilerin online platformlarda yaptığı alışveriş sayısı offline platformlarda yaptığı alışveriş sayısından çok daha fazladır.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 184;ortanca değeri 140 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 985;ortanca değeri 779 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  ödediği toplam ücret değişkeni  için hem ortalama hem de median değerlerinde çok önemli bir fark gözlenmektedir.
>> Müşterilerin online platformlarda şirkete bıraktığı para offline platformlarda şirkete bıraktığı paradan çok daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en kötü olan 2. segment. (1,7)
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(943) ve median(714)'larını incelediğimizde bu segment flo için en eski müşterilerin bulunduğu ikinci segmenttir.(6,7)

Özet: Online platformlarda alışveriş yapma ve şirkete para bırakma potansiyeli offline platformlara göre çok daha yüksektir. Alışveriş yapma sıklığı en kötü olan 2. segmenttir. Fakat bu müşteriler zamanında flo'dan çok sık alışveriş yapmış ve şirkete önemli ölçüde para bırakmışlardır. Bu müşterilerin peşine düşmeliyiz. :)  En eski müşterilerin bulunduğu ikinci segmenttir.

Aksiyon: bu müşterilerle kişisel olarak iletişime geçip yeni teklifler sunulabilir.
"""


###############################################################
# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
###############################################################

# 1. Görev 2'de standarlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
hc_complete = linkage(model_df, 'ward')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=11.70, color='r', linestyle='--')
plt.show()


# 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
hc = AgglomerativeClustering(n_clusters=6)
segments = hc.fit_predict(model_df)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()

final_df["segment"].value_counts()

# 3. Herbir segmenti istatistiksel olarak inceleyeniz.


final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","median","count"],
                                  "order_num_total_ever_offline":["mean","median","count"],
                                  "customer_value_total_ever_offline":["mean","median","count"],
                                  "customer_value_total_ever_online":["mean","median","count"],
                                  "recency":["mean","median","count"],
                                  "tenure":["mean","median","count"]})





final_df[final_df["segment"]]


#satır 350 dan wardı silince ve 364. satırda n_clusterı 5 yapınca ki yorumlar. Business kararlar alırken yardımcı olması için paylaşılmıştır.
"""
Segment-1:  Bu segmentte 8747 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.37;ortanca değeri 1 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.99;ortanca değeri 2 olarak hesaplanmıştır.
>> Müşterilerin offline platformlarda yaptığı alışveriş sayısı online platformlarda yaptığı alışveriş sayısından fazladır diyebiliriz.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 261;ortanca değeri 189 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 208;ortanca değeri 168 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  para bırakma potansiyeli hem ortalama hem de median değerlerinde  önemli  bir fark gözlenmektedir.

>> Müşterilerin offline platformlarda  şirkete para bırakma potansiyeli  online platformlardakinden daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde bu segmente  alışveriş yapma sıklığı en kötü olan ikinci segmenttir diyebiliriz.
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(569) ve median(461)'larını incelediğimizde bu segment flo için diğer segmentlerle kıyaslandığında en yeni sayılabilecek ikinci segmenttir diyebiliriz.

Özet: Offline plaftormlarda ödedikleri para online'a göre daha fazla. Doğru kampanyalarla alışveriş yapma sıklığı arttırılabilir.

"""

"""
Segment-2:  Bu segmentte 4994 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 3.75;ortanca değeri 2 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.82;ortanca değeri 1 olarak hesaplanmıştır.
Offline platformlarda alışveriş sayısı düşük. Online platformu tercih ediyorlarlar.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 240;ortanca değeri 165 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 638;ortanca değeri 407 olarak hesaplanmıştır.
Müşterilerin online platformlarda para bırakma potansiyeli offline platformlardakinden çok daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en iyi olan müşteriler bu segmenttedir.
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(715) ve median(608)'larını incelediğimizde bu segment flo için en yeni müşterilerin bulunduğu üçüncü segmenttir.

Özet: Online plaftormlarda ödedikleri para offline'a göre daha fazla. Flo'dan sıklıkla alışveriş yapıyorlar. Flo için en yeni müşterilerin bulunduğu üçüncü segmenttir. Doğru kampanyalarla alışveriş yapma sıklığı ver para bırakma potansiyelleri arttırılabilir.

"""

"""
Segment-3:  Bu segmentte 2396 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 6.91;ortanca değeri 5 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 3.08;ortanca değeri 3 olarak hesaplanmıştır.
>> Müşterilerin online platformlarda yaptığı alışveriş sayısı offline platformlarda yaptığı alışveriş sayısından çok daha fazladır.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 427;ortanca değeri 352 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 1104;ortanca değeri 745 olarak hesaplanmıştır.
Müşterilerin online ve offline platformlarda  ödediği toplam ücret değişkeni için hem ortalama hem de median değerlerinde önemli  bir fark gözlenmektedir.
>> Müşterilerin online platformlarda para bırakma potansiyeli offline platformlardakinden çok daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en az olan 3. segment.
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(914) ve median(682)'larını incelediğimizde bu segment flo için en eski müşterilerin olduğu segmentlerden biridir.

Özet: online platformlarda alışveriş yapma ve para bırakma eğilimleri en yüksek olan segment.  Alışveriş yapma sıklığı en az olan 3. segment. (Arttırılmalıdır. Potansiyel fazla. )

"""

"""
Segment-4:  Bu segmentte 985 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.40;ortanca değeri 1 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.20;ortanca değeri 1 olarak hesaplanmıştır.
Online platformlarda alışveriş yapma sayıları diğerine göre az bir farkla da olsa yüksek.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 200;ortanca değeri 160 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 227;ortanca değeri 181 olarak hesaplanmıştır.
>> Müşterilerin online platformlarda para bırakma potansiyeli offline platformlardakinden biraz daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en fazla olan 2. segment.
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(80) ve median(51)'larını incelediğimizde bu segment flo için en yeni müşterilerin bulunduğu segmenttir.

Özet: Online platformlarda alışveriş yapma ve şirkete para bırakma potansiyeli offline platformlara göre biraz daha yüksektir. Alışveriş yapma sıklığı en iyi olan 2. segment olmasına rağmen alışveriş sayısı ve para bırakma potansiyeli açısından diğer segmentlerle kıyaslandığında az sayılabilcek düzeydedir.  En yeni müşteriler bu segmenttedir.

"""

"""
Segment-5:  Bu segmentte 2823 müşteri vardır.
Müşterinin online platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 4.74;ortanca değeri 4 olarak hesaplanmıştır.
Müşterinin offline platformda yaptığı toplam alışveriş sayısı değişkeninin ortalama değeri 1.12;ortanca değeri 1 olarak hesaplanmıştır.
Online platformlarda alışveriş yapma sayıları fazladır.

Müşterilerin offline alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 123;ortanca değeri 99 olarak hesaplanmıştır.
Müşterilerin online alışverişlerde ödediği toplam ücret değişkeninin ortalama değeri 722;ortanca değeri 591 olarak hesaplanmıştır.
>> Müşterilerin online platformlarda para bırakma potansiyeli daha fazladır.

Bu segment için müşterilerin recency(son alışverişten bu yana kadar geçen zaman)  değerlerinin mean ve median'larını incelediğimizde alışveriş yapma sıklığı en az olan segment.
Bu segment için müşterilerin tenure(last order date-first order date) değerlerinin mean(876) ve median(650)'larını incelediğimizde bu segment flo için en eski müşterilerin bulunduğu segmentler arasındadır.

Özet: Online platformlarda alışveriş yapma ve şirkete para bırakma potansiyeli offline platformlara göre daha yüksektir. Alışveriş yapma sıklığı en az olan segmenttir. Fakat şirkete online platformlarda geçmişte güzel paralar burakmıştır. Doğru kampanyalarla bu müşterilerin yeniden dönüşü sağlanabilir.

"""