# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:48:47 2022

@author: MERYEM
"""

import pandas as pd #Pandas Kütüphanesi
import matplotlib.pyplot as plt #Pca Grafiği için gerekli olan kütüphane
from sklearn.cluster import KMeans #Kmeans ile kümeleme yapmak için gerekli kütüphane
import numpy as np #Numpy kütüphanesi
from sklearn.preprocessing import MinMaxScaler #Normalizasyon için gerekli kütüphane


geciciVeri = pd.read_csv("../final/Final-data.txt", sep=',', names= ['Sports','Religious','Nature','Theatre','Shopping','Picnic'])

"""
Burada elimizdeki Txt dosyanını pandas kütüphanesi yardımıyla okuttuk. Txt Dosyasında verilerin kolon olarak ve birbirinden ayrılması için kullanılan
 "," ayracını tanıttık. Böylece program her "," gördüğünde oraya kadar olan veriyi kaydedip diğer kolona geçecek.
names kullanarak'ta Tüm kolon isimlerini tanıttık.
"""

veri = geciciVeri[1:] #Pandas'ta veriler 0. İndisten başlar Bu satırda 1.İndisten itibaren kirli veriyi veriye aktardığımızda Text 
#dosyasındaki kolon isimlerini veri olarak okumamış oluyoruz.

verinp = np.array(veri) #Elimizdeki veriyi aynı zamanda Numpy formatına dönüştürdük





"""
Bu Kısımı çalıştırdım ve çıkan sonuçları Sunuma ekledim. Yeşil kısımlar ve # ile belirttiğim alanlar bilgi verme amaçlıdır.
K-Means Kümeleme yöntemini burada kendi oluşturduğum örnekten yararlanarak istediğiniz şekle getirdim
"""

"""
plt.scatter(veri['Shopping'],veri['Picnic']) #Shopping ve Picnic Kolonlarının Pca Grafiğini çıkardım
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

Shopping_Picnic = veri.iloc[:,4:6] # Yukarıda direkt olarak Plotunu gösterdiğimiz kolonları ayrıştırarak ayrı bir Dataframe oluşturdum

kmeans = KMeans(2) #Bu satırda from sklearn.cluster import KMeans kütüphanesini kullanarak ve WCSS Tablosundan yararlanarak k=2 verdik
kmeans.fit(Shopping_Picnic) #Shopping ve Picnik kolonlarında .fit kullanarak örnekleri kümeleme için eğitiyoruz
#yani uygunluklarına bakarak kümeleme yapılıyor ve kümeleme için hazır hale getiriliyor.

kumeleme_değerleri = kmeans.fit
_predict(Shopping_Picnic) # Son kümeleme işlemi olarakta fit_predick kullanarak
#Küme merkezlerini hesaplatıp,her örnek için küme indeksini tahmin ettiriyorum. Training kısmı da deniebilir

kumelenmis_veri = veri.copy() #Verinin orjinalini kopyalıyoruz.
kumelenmis_veri['kumeler'] = kumeleme_değerleri #Öğrendiğimiz kümeleme değerlerini veri'mize dahil ediyoruz.
plt.scatter(kumelenmis_veri['Shopping'],kumelenmis_veri['Picnic'],c=kumelenmis_veri['kumeler'],cmap='rainbow')
#Son olarakta Shopping ve Picnik Kolonlarının Öğrendiğimiz Kümeleme yöntemiyle birlikte Rainbow Renk komutu ile Plot yani PCA oluşturuyoruz.
"""

"""Yukarıda açıkladığım yöntemler ve verilerden faydalanarak Proje'nin gereksinimlerini yerine getiricem.
 Umarım beğenirsiniz :D
"""

gorsel = input("Veri Görselleştirme yapmak istiyor musunuz? YES/NO")# K değerini str olarak kullanıcıdan aldık 

yes= "YES"
f = open("sonuc.txt", "a")
if gorsel == yes:

    i = 0
    i = input("Lütfen K değerini Giriniz")# K değerini str olarak kullanıcıdan aldık 
    k=int(i) #Str değerini KMeans'e ekleyebilmek için İnt değerine dönüştürdük
    kmeans = KMeans( k ) #Kullanıcıdan aldığımız değeri ekledik
    

    birincideg = input("İlk değişkenin ismini giriniz.Örnek: 'Sports','Religious','Nature','Theatre','Shopping','Picnic' = ")# K değerini str olarak kullanıcıdan aldık
    ikincideg = input("ikinci değişkenin ismini giriniz Örnek: 'Sports','Religious','Nature','Theatre','Shopping','Picnic' = ")# K değerini str olarak kullanıcıdan aldık
    #Birinci kolon ve ikinci kolonun isimlerini kullanıcıdan alıyoruz
    

    kmeans.fit(veri) #Bu satırları yukarıda açıkladığım için tekrar açıklama gereği duymadım.
    
    kumeleme_değerleri = kmeans.fit_predict(veri)
    
    kumelenmis_veri = veri.copy() 
    kumelenmis_veri['kumeler'] = kumeleme_değerleri #Öğrendiğimiz kümeleme değerlerini veri'mize dahil ediyoruz.
    
    kumelistesi = list(kumeleme_değerleri) 
 
    kümeyazi= str("küme")
    
    plt.scatter(kumelenmis_veri[birincideg],kumelenmis_veri[ikincideg],c=kumelenmis_veri['kumeler'],cmap='rainbow')
    #Kullanıcıdan aldığımız kolon isimlerini kullanarak kümeleme grafiğimizi gösteriyoruz.
    plt.show()

    
    sifirsay= 0
    birsay= 0
    ikisay= 0
    ucsay= 0
    dortsay= 0
    bessay= 0
    altisay= 0
    yedisay = 0
    sekizsay = 0
    for element in kumelistesi:   
         f.write("\n" + "Küme = " )
         f.write(str(element) )
         
         if element == 0:
            sifirsay = sifirsay+1
         elif element == 1:
             birsay= birsay + 1
         elif element == 2:
             ikisay= ikisay + 1
         elif element == 3:
             ucsay= ucsay + 1    
         elif element == 4:
              dortsay= dortsay + 1   
         elif element == 5:
             bessay= bessay + 1
         elif element == 6:
             altisay= altisay +1
         elif element == 7:
             yedisay= yedisay +1   
         elif element == 8:
             sekizsay= sekizsay +1     
             
if gorsel == yes:             
    f.write("\n" + "Kume 0 = " + str(sifirsay) + " Kayit  " +"\n" + " Kume 1 = " + str(birsay) + " Kayit  " +"\n" +" Kume 2 = " + str(ikisay)  + " Kayit  " +"\n" +" Kume 3 = " + str(ucsay) + " Kayit  " +"\n" +" Kume 4 = " + str(dortsay) + " Kayit  " +"\n" +" Kume 5 = " + str(bessay) + " Kayit  " +"\n" +"Kume 6 = " + str(altisay) + " Kayit  " +"\n" +"Kume 7 = " + str(yedisay) +" Kayit  " +"\n" + "Kume 8 = " + str(sekizsay) + " Kayit  " +"\n" )   


wcss = []
kume_sayisi_listesi = range(1, 6) #Verisetindeki tüm kolonlarda gezdiriyoruz
for i in kume_sayisi_listesi :
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(veri)
    wcss.append(kmeans.inertia_)

plt.plot(kume_sayisi_listesi, wcss)
plt.title('Küme sayisi Belirlemek için Dirsek Yöntemi')
plt.xlabel('Küme sayisi')
plt.ylabel('WCSS')
plt.show()
OrtWCSS = sum(wcss) #Tüm wcss değerlerinin ortalaması
f.write( "WCSS degerleri = " + str(wcss) + "\n")
f.write( "WCSS Ortalamasi = " + str(OrtWCSS) + "\n")


" BCSS için Verimizin önce Minmaxını alıyoruz sonrada Normalizasyon işlemi yapıyoruz. 0-1 Arasına çekiyoruz tüm değerleri"
scaler = MinMaxScaler()
norm_veri = veri.copy()
def minmaxscaler(x):
    for columnName, columnData in x.iteritems():
        x[columnName] = scaler.fit_transform(np.array(columnData).reshape(-1, 1))
    
minmaxscaler(norm_veri)
norm_veri.head()

k = list(range(2,8)) #2 Den 8 e kadar k değerlerinde gezdiriyoruz
between_clusters_sum_of_squares = [] 
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(norm_veri)
    between_clusters_sum_of_squares.append(kmeans.inertia_)
    
OrtBCSS = sum(between_clusters_sum_of_squares) #Tüm wcss değerlerinin orlaması

f.write( "BCSS degerleri= " + str(between_clusters_sum_of_squares) + "\n")
f.write( "BCSS Ortalamasi = " + str(OrtBCSS) + "\n")

# Function to plot initial data points.
def plot_data_before(data):
    x, y = data.iloc[:, 0] , data.iloc[:, 1]
    plt.scatter(x, y, color = "m",marker = "o", s = 30)

    plt.xlabel('Sports', size = 20)
    plt.ylabel('Religious', size = 20)

    plt.show()
    

# Function to find points that are closest to centriods
def closestCentriods(data, ini_cent):
    
    K = np.shape(ini_cent)[0]
    
    m = np.shape(data)[0]
    idx = np.zeros((m, 1))
    
    cent_vals = np.zeros((m, K))
    # Subtract each data row with each centroid value and get the different
    # Find sqaured sum of different of eache each row
    for i in range(K):
        Diff = data - ini_cent[i,:]
        cent_vals[:, i] = np.sum(Diff**2, axis = 1)
    
    # Return index of minimum value column wise.
    idx =  cent_vals.argmin(axis=1)
    return idx


# Function to find/update centriod, mean of a cluster
def calcCentriods(data, idx, K):
    
    n = np.shape(data)[1]
    centriods = np.zeros((K, n))
    for i in range(K):
        x_indx = [wx for wx, val in enumerate(idx) if val == i]
        centriods[i, :] = np.mean(data[x_indx, :], axis= 0)
        #print('mean:', np.mean(data[x_indx, :], axis= 0))  
    return centriods


# Function to iterate centriod search, 
# until we have found optimum result
def applyKmeans(data, initial_centroids, max_iters):
    
    dataArr = np.array(data)
    K = np.shape(initial_centroids)[0]
    centriods = initial_centroids
    previous_centriods = centriods
    
    for i in range(max_iters):
        print('\n\n K-Means iteration number {}'.format(i+1))
        
        # Find the closest centriods
        idx = closestCentriods(dataArr, centriods)
        centriods = calcCentriods(dataArr, idx, K)
        
        
        print('\n\n The centriods are')
        print('\n The value of first centriod : {} \n The value of second centriod {}\n\n'.format(centriods[0, :], centriods[1, :]))
        
        plot_data_after(data, centriods)
        

        # Break when centriod doesn't change
        if np.equal(previous_centriods, centriods).all():
            break
        else:
            previous_centriods = centriods
        
# Scatter centiods along with initial data set.      
def plot_data_after(data, centriods):
    x, y = data.iloc[:, 0] , data.iloc[:, 1]
    cx, cy = centriods[:, 0], centriods[:, 1]
    plt.scatter(x, y, color = "m",marker = "o", s = 30)

    plt.xlabel('Sports', size = 20)
    plt.ylabel('Religious', size = 20)
    
    plt.scatter(cx, cy, color = "b",marker = "*", s = 300)

    plt.show()
    
    Dunn = cx, cy
    f.write( "Dunn index = " + str(Dunn) + "\n")
    
        
# Main function to fetch data set, and apply K-means    
def main():
    
    df = pd.read_csv("../final/Final-data.txt")
    data = df[['Sports', 'Religious']] 
    plot_data_before(data)
    initial_centroids = np.array([[1, 6], [0, 7]])
    
    dataArr = np.array(data)
    idx = closestCentriods(dataArr, initial_centroids)
    
    K = np.shape(initial_centroids)[0]
    centriods = calcCentriods(dataArr, idx, K)
    
    plot_data_after (data, centriods) 
    
    applyKmeans(data, initial_centroids, 10)
    
    
if __name__ == '__main__':
    main()

f.close()
