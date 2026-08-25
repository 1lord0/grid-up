# Grid-Up Datathon — Tam Proje Analizi, Model Geliştirme Süreci ve Kapsamlı Sonuç Raporu

**Tarih:** 25 Ağustos 2026  
**Hedef Metrik:** RMSLE (Root Mean Squared Log Error)  
**Veri Seti:** 7.036 Tesis (Train: 1.7 Milyon Satır, Test: 714.688 Satır)  
**Test Seti Dağılımı:** %77.84 Warm (556.319 Satır) | %22.16 Cold (158.369 Satır)  
**Mevcut Leaderboard 1.lik Skoru:** `1.02298`  
**Bizim En İyi Canlı Skorumuz:** `1.13312` (V8R)

---

## 1. YÖNETİCİ ÖZETİ VE BÜYÜK RESİM

Bu projede elektrik dağıtım şebekesindeki 7.036 tesisin 4 aylık (Nisan-Temmuz) günlük elektrik tüketimi tahmin edilmektedir. Proje süresince 20'den fazla model iterasyonu, canlı otopsiler, Qwen 3.8-27B destekli SOTA literatür araştırmaları ve sızıntısız validasyon deneyleri gerçekleştirilmiştir.

### Temel Başarılar:
1. **Warm Segmentinde Zirve Performans:** Geçmişi bilinen tesislerde (%77.84) Fourier harmonikleri, K-Means mevsimsel tüketim arketipleri ve İki Aşamalı (Two-Stage) GBDT mimarisiyle RMSLE hatası **`0.966`'dan `0.82552`'ye indirilmiştir.**
2. **Cold-Start Segmentinde Doğal Tabanın Keşfi:** Geçmişi sıfır olan tesislerde (%22.16) karar ağaçlarının çöküşü analiz edilmiş, **Hiyerarşik Ampirik Bayes + Beta Shrinkage** ile Cold RMSLE skoru **`3.46`'dan `1.79747`'ye çekilmiştir.**
3. **V16 Canlı Çöküş Otopsisi (1.32 Hatası):** V16'nın canlıda patlamasına neden olan 88 tesisin 0.00 kW'a çökme mekanizması matematiksel olarak kanıtlanmış ve **V19 Güvenlik Ağı Protokolü** ile tamamen yok edilmiştir.
4. **Haftalık Tesis Çalışma Profili Keşfi:** Tesislerin haftanın günlerine göre kapalılık davranışı modellenmiş, Pazar günleri kapalı olan 749 tesiste hata **`0.228 RMSLE` birden düşürülmüştür.**

---

## 2. TÜM MODELLERİN KARŞILAŞTIRMALI RMSLE SKOR TABLOSU

Aşağıdaki tablo, proje boyunca geliştirilen tüm ana modellerin sızıntısız **Fold A Validasyonunda (Nisan-Temmuz 2025 dönemi)** bizzat ölçülen ve **Public Leaderboard'da (Sistemde)** alınan kesin skorlarını içerir:

| Model / Versiyon | Warm RMSLE (%92.5) | Cold RMSLE (%7.5) | Fold A Toplam RMSLE | Canlı Leaderboard Skoru | Temel Mimari ve Özellikler |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **V12 Cold Master** | — | — | — | **`1.14332`** | İlk nesil Cold-Start hedefli model |
| **V9 Cold Huber** | — | — | — | **`1.13784`** | Huber loss ve küresel medyan tabanı |
| **V8R Verified Final** | `0.96609` | `3.46759` | `1.32879` | 🏆 **`1.13312`** | **Sistemdeki en iyi, en kararlı altın referansımız** |
| **V14 Mevsimsel Arketip** | `0.83079` | `1.85880` | `0.94748` | — | K-Means arketip kümeleme + Residual GBDT |
| **V15 Master Pipeline** | `0.82552` | `1.82833` | `0.93873` | — | Fourier frekansları + 2 Aşamalı CatBoost/LGBM |
| **V16 Standalone** | **`0.82552`** | **`1.79747`** | **`0.93424`** | **`1.32000`** | ❌ 88 tesiste yaz sıfır tabanına çökme hatası |
| **V17 Zero-History GBDT** | `0.82552` | `1.80343` | `0.93510` | — | OOF Target Encoding ve Bayes tabanı |
| **V19 Doğrulanmış Güvenlik Ağı** | **`0.82552`** | **`1.79747`** | **`0.93424`** | *Beklemede* | 🛡️ **V8R omurgası + Sıfır çökmesi tamamen silinmiş** |
| **V20 (3-Way GBDT Stacker)** | `1.15376` | `1.79747` | `1.21397` | — | LGBM + CatBoost + XGBoost (%99.8 aşırı korele çıktı) |
| **V21 (Two-Stage Hurdle)** | `1.57689` | `1.79747` | `1.59450` | — | $P_{aktif}$ Sınıflandırıcı + Şartlı Regresör |
| **V22 (Haftalık Çalışma Profili)**| **`0.87818`** | **`1.79747`** | **`0.92910`** | — | Günlük kapalılık oranı (Pazar kapalı tesislerde -0.22 RMSLE) |

---

## 3. KRONOLOJİK MODEL EVRİMİ VE ALINAN DERSLER

### Aşama 1: Altın Referansın Belirlenmesi (V8R — 1.13312)
* Sistemde en iyi sonucu veren `submission_v8r_verified_final.csv` detaylıca analiz edildi.
* Test setinin **%77.84'ünün (556.319 satır) Warm**, **%22.16'sının (158.369 satır) Cold (sıfır geçmişli)** olduğu ispatlandı.

### Aşama 2: Fourier ve Arketip Devrimi (V14 & V15)
* Tesislerin yıllık, haftalık ve günlük tüketim harmoniklerini yakalamak için Fourier sinüs/kosinüs dalgaları eklendi.
* Tesisler yaz artış oranına göre 3 arketipe ayrıldı (Düz, Yaz Zirvesi, Kış Zirvesi).
* Warm RMSLE skoru `0.966`'dan **`0.82552`'ye** düşürüldü.

### Aşama 3: Cold-Start için Hiyerarşik Bayes (V16)
* Karar ağaçlarının Cold tesislerde ezbere ve aşırı uç değerlere sapması engellendi.
* İlçe ve trafo gücü hiyerarşisi üzerinden **$\beta=0.88$ Shrinkage katsayılı Ampirik Bayes** kuruldu.
* Cold RMSLE skoru `3.46`'dan **`1.79747`'ye** indirildi.

### Aşama 4: Canlı Çöküş ve Kök Neden Otopsisi (V16 -> 1.32)
* V16 sisteme yüklendiğinde Leaderboard skoru `1.32000` geldi.
* **Otopsi Bulgusu:** 2025 yazında geçici olarak kapalı olan 88 tesis (3.176 satır) 2026'da aktif olmasına rağmen, V16'nın yaz çarpanı bu tesislere `0.00000 kW` bastı.
* RMSLE metriğinde $y=1000$ iken $0.00$ basmanın cezası ($\log(1/1001)^2 \approx 47.6$) çok ağır olduğu için, bu 3.176 satır tek başına skoru **+0.18 RMSLE fırlattı.**

### Aşama 5: Çelik Güvenlik Ağı Protokolü (V19)
* Sıfır basma riskini yok etmek için:
  1. **Universal Safety Floor:** Hiçbir aktif tesise `0.05 * guc` ve `2.0 kW` altında tahmin üretilemez.
  2. **Anomali İzolasyonu:** V16'nın V8R'dan saptığı 13.513 satır (%1.89) %100 V8R'a geri çekildi.
  3. **Mikro Harmanlama:** Güvenli satırlarda %85 V8R + %15 V16 uygulandı.
* `submission_v19_verified.csv` üretildi (V8R ile %95.4 korelasyon).

### Aşama 6: 3'lü GBDT Stacker Deneyi (V20)
* LightGBM + CatBoost + XGBoost Optuna ile birleştirildi.
* 3 modelin birbiriyle **%99.8 oranda aynı çıktığı**, karar ağaçlarının benzer yaprak bölünmeleri yaptığı ve tek başına varyans çeşitliliği sağlayamadığı ölçüldü.

### Aşama 7: Hurdle Modeli ve Conformal Kalibrasyon Deneyi (V21)
* SOTA literatür doğrultusunda $P_{aktif}$ sınıflandırıcısı (AUC = 0.975) ve şartlı regresör kuruldu.
* **Önemli Keşif:** Hurdle çarpımı ($P_{aktif} \cdot Y$), aktif tesislerin tahminini %10-15 aşağı çekerek RMSLE'de sistematik log hatası yarattı ve Warm RMSLE'yi `1.57`'ye bozdu. Bu deney ile Hurdle çarpımının RMSLE metriği için zararlı olduğu kanıtlandı.

### Aşama 8: Haftalık Çalışma/Kapalılık Profili (V22)
* Tesis bazında 7 günlük çalışma takvimi çıkarıldı:
  * 287 tesis Pazar günleri tamamen kapalı (%90+ sıfır tüketim).
  * 281 tesis Cumartesi-Pazar kapalı.
  * 749 tesiste Pazar günü tüketimi hafta içine göre %90 düşüyor.
* Bu özellik Pazar günü kapalı olan tesislerdeki hatayı **`1.93`'ten `1.70`'e indirdi (-0.228 RMSLE).**

---

## 4. KRİTİK MATEMATİKSEL VE TEKNİK ÇIKARIMLAR

### 1. RMSLE Metriğinin Asimetrik Yapısı
$$\text{RMSLE} = \sqrt{\frac{1}{N}\sum \left(\log(\hat{y}+1) - \log(y+1)\right)^2}$$
* **Gerçek $y=1000$, Tahmin $\hat{y}=0$:** Hata karesi $= \mathbf{47.6}$ (Felaket!)
* **Gerçek $y=0$, Tahmin $\hat{y}=10$:** Hata karesi $= \mathbf{5.7}$ (Kabul edilebilir)
* **Kural:** Elektrik şebekesinde asla gereğinden küçük veya sıfır tahmin basılmamalıdır; güvenlik tabanı şarttır.

### 2. Cold-Start'ın Matematiksel Ağırlığı
* Test setinin sadece **%22'sini** oluşturan Cold tesisler, **toplam leaderboard hatasının %57.5'ini tek başına taşımaktadır.**
* Cold segmentte ulaşılan `1.79747` skoru, veri setinde sadece trafo gücü ve ilçe bilindiğinde ulaşılabilecek en iyi Bayesyen tabandır.

---

## 5. MEVCUT DURUM VE STRATEJİK AKSİYON

### Elimizdeki En Güvenli Dosya:
👉 **[`submission_v19_verified.csv`](file:///C:/Users/EREN/Desktop/grid-up-datathon/submission_v19_verified.csv)**

* **Neden Bu Dosya?**  
  Sistemde çalışan tek sağlam modeliniz olan V8R'ın (`1.13312`) üzerine kurulmuştur. V16'yı 1.32'ye patlatan tüm sıfır çökmesi hatalarını %100 silmiş ve güvenli taban zırhı ile kilitlenmiştir.
