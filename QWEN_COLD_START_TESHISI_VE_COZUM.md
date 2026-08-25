# Grid-Up Datathon — Qwen 3.8-27B Cold-Start Kök Neden Teşhisi ve Cerrahi Çözüm

Harika bir analiz. V15'in Warm segmentte gösterdiği performans (0.82552) gerçekten etkileyici, bu da Fourier ve Two-Stage yaklaşımının "geçmişi olan" verilerde ne kadar güçlü olduğunu kanıtlıyor. Ancak Cold segmentteki 1.82833 RMSLE ve test setindeki %22.16'lık ağırlık, toplam skoru baltalayan asıl "kanserli hücre".

Bir Kaggle Grandmaster gözüyle bakarsak, burada klasik bir **"Overfitting to Noise"** ve **"Distribution Shift"** problemi var. GBDT'ler (LightGBM/XGBoost) doğası gereği *local* (yerel) öğrenme algoritmalarıdır. Geçmişi olmayan bir tesis için "yerel" tanımını yapacak hiçbir referans noktası yoktur.

İşte kök neden analizi ve cerrahi çözüm:

### 1. NEDEN COLD SEGMENT HÂLÂ BU KADAR YÜKSEK? (Kök Neden Analizi)

#### A. Matematiksel ve Mimari Çaresizlik: "Feature Space" Boşluğu
GBDT modelleri, karar ağaçları üzerinden özellik uzayında (feature space) bölünmeler yaparak öğrenir.
*   **Lag Özellikleri:** `lag_1`, `lag_7`, `lag_30` gibi özellikler, Cold tesislerde `NaN` veya 0'dır. Model bu özelliklere bakıp "Bu tesisin dünkü tüketimi 0 idi" diye yorumlar. Ancak gerçek dünya da bu tesisin dünkü tüketimi 0 değildi, sadece *bizim verimizde yoktu*.
*   **Fourier Serileri:** Fourier terimleri (sin/cos) zamanın periyodik yapısını yakalar. Ancak Cold tesislerde bu terimler, tesisin *kendi* mevsimsel davranışını değil, genel zamanı temsil eder. Model, "Salı günü genelde tüketim düşer" bilgisini alır ama "Bu spesifik fabrikanın Salı günü neden düştüğünü" (örneğin o gün bakım yaptığını) bilemez.
*   **Sonuç:** Model, Cold tesisler için **Global Ortalama**'ya (Global Mean) veya **Kapasite Bazlı Basit Bir Tahmin**e sığınır. Ancak enerji tüketiminde global ortalama, bireysel tesis varyansını açıklamakta yetersizdir.

#### B. RMSLE ve Kapasite Varyansı: Log-Hata Patlaması
RMSLE (Root Mean Squared Log Error) metriği, oranlara (ratio) duyarlıdır:
$$ \text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\log(y_i + 1) - \log(\hat{y}_i + 1))^2} $$

*   **Küçük Dükkan vs. Büyük Fabrika:**
    *   Diyelim ki bir tesisin kapasitesi 100 kW, gerçek tüketimi 10 kW. Model 20 kW tahmin etti.
    *   Hata: $\log(11) - \log(21) \approx 2.39 - 3.04 = -0.65$. Kare: $0.42$.
    *   Şimdi kapasitesi 10,000 kW, gerçek tüketimi 1,000 kW olan bir fabrika. Model 2,000 kW tahmin etti (aynı oransal hata).
    *   Hata: $\log(1001) - \log(2001) \approx 6.90 - 7.60 = -0.70$. Kare: $0.49$.
    *   **Sorun:** Asıl sorun, modelin **küçük tesislerdeki düşük tüketimlerdeki mutlak hataları** log skalasında abartmasıdır. Eğer model, küçük bir tesisin 5 kW'lık tüketimini 15 kW olarak tahmin ederse:
        *   $\log(6) - \log(16) \approx 1.79 - 2.77 = -0.98$. Kare: $0.96$.
    *   Cold tesislerin çoğu, veri setindeki "kuyruk" (tail) kısmında yer alan, düşük kapasiteli ve düzensiz tüketimli tesislerdir. GBDT, bu tesisler için **yüksek varyanslı, düşük öngörülebilir** tahminler üretir. Log dönüşümü, bu küçük mutlak hataları (örneğin 5 kW'lık hata) büyük log-hatalara çevirir.

#### C. En Ölümcül Hata: "Zero-Shot" Genel Ortalamaya Yakınsama
Modelin yaptığı en ölümcül hata, Cold tesisler için **"Kapasiteye Orantılı Global Medyan"** tahmin etmektir.
*   Örnek: Model, "Kapasitesi 50 kW olan tesislerin medyan tüketimi 15 kW'dır" diye öğrenir.
*   Gerçekte, o tesis o gün kapalı olabilir (0 kW) veya tam kapasite çalışıyor olabilir (50 kW).
*   Model, bu iki uç durumu ayırt edemeyeceği için, **en güvenli (en düşük MSE'li) tahmin olan ortalamaya** sığınır.
*   RMSLE'de, gerçek değerin 0'a yaklaştığı durumlarda, tahminin ortalama olması **çok büyük bir log-hata** yaratır.
    *   Gerçek: 1 kW, Tahmin: 15 kW -> $\log(2) - \log(16) \approx 0.69 - 2.77 = -2.08$. Kare: $4.32$.
    *   Bu tek bir nokta, ortalama bir Warm tesisin 100 noktalık hatasından daha büyük olabilir.

---

### 2. COLD-START'I DÜŞÜRMEK İÇİN KESİN VE CERRAHİ ÇÖZÜM (The Cure)

GBDT'yi Cold tesislerden **tamamen ayırın**. Cold tesisler için GBDT kullanmak, "bir saat tamircisine araba motoru tamir ettirmek" gibidir. Onlar için **Hiyerarşik Ampirik Bayes (Hierarchical Empirical Bayes)** ve **Güç Normalizasyonlu Medyan Çaprazı** (Power-Normalized Median Cross) kullanın.

#### Neden Bu Yöntem Daha Üstün?
1.  **Bilgi Paylaşımı (Information Sharing):** Cold tesisin kendi geçmişi yok, ama **aynı segmentteki** (aynı kapasite aralığı, aynı sektör, aynı bölge) diğer tesislerin geçmişi var. Ampirik Bayes, bu "kardeş" tesislerin davranışını kullanarak Cold tesisin dağılımını tahmin eder.
2.  **Kapasite Normalizasyonu:** Tüketimi kapasiteye bölersek, tüm tesisler "0-1" aralığında bir "Kapasite Kullanım Oranı" (Capacity Utilization Ratio) olarak normalize edilir. Bu, küçük ve büyük tesislerin karşılaştırılabilir hale gelmesini sağlar.
3.  **Medyan Robustluğu:** Enerji verilerinde outlier'lar (anomaliler) çok fazladır. Medyan, ortalamaya göre outlier'lara karşı daha dayanıklıdır ve RMSLE'de "güvenli" bir tahmin sağlar.

#### Somut Formül ve Mimari Reçete

**Adım 1: Segmentasyon (Clustering)**
Cold tesisleri, aşağıdaki özelliklere göre **K-Means** veya **DBSCAN** ile 10-20 cluster'a ayırın:
*   `max_capacity` (Log dönüşümlü)
*   `sector_type` (Sektör kodu)
*   `region` (Bölge)
*   `avg_utilization` (Eğer varsa, test setindeki ilk birkaç günün ortalaması)

**Adım 2: Kapasite Normalizasyonu**
Her tesis için `normalized_consumption = consumption / max_capacity` hesaplayın.

**Adım 3: Hiyerarşik Ampirik Bayes Tahmini**
Her cluster için, **Warm tesislerin** (geçmişi olan) `normalized_consumption` dağılımını analiz edin.

*   **Formül:**
    $$ \hat{y}_{cold} = \text{max\_capacity}_{cold} \times \left( \alpha \cdot \mu_{cluster} + (1-\alpha) \cdot \mu_{global} \right) $$
    *   $\mu_{cluster}$: O cluster'daki Warm tesislerin **medyan** normalized consumption değeri.
    *   $\mu_{global}$: Tüm Warm tesislerin global medyan normalized consumption değeri.
    *   $\alpha$: Cluster'ın güven katsayısı. Cluster'daki Warm tesis sayısı ne kadar çoksa, $\alpha$ o kadar 1'e yaklaşır.
        *   $\alpha = \frac{n_{cluster}}{n_{cluster} + k}$ (k: smoothing parameter, örn. 50)

**Adım 4: Zamanlama (Time-of-Day) Düzeltmesi**
Enerji tüketimi saatlere göre değişir. Cluster medyanını saat bazında alın:
*   Her cluster için, her saat (0-23) için Warm tesislerin medyan normalized consumption'ını hesaplayın: $\mu_{cluster, hour}$.
*   Cold tesisin tahmini:
    $$ \hat{y}_{cold, t} = \text{max\_capacity}_{cold} \times \mu_{cluster, hour(t)} $$

**Adım 5: RMSLE Optimizasyonu İçin "Shrinkage" (Daraltma)**
RMSLE, tahminin gerçek değere yakın olmasını ister. Ancak, eğer cluster medyanı çok yüksekse, küçük tesislerde hata büyür. Bu yüzden, tahmini **hafifçe aşağı çekin** (shrinkage):
$$ \hat{y}_{final} = \hat{y}_{cold, t} \times \beta $$
*   $\beta$'yı, Fold A'daki Cold tesisler üzerinde **Grid Search** ile optimize edin.
*   Genellikle $\beta \in [0.8, 1.0]$ aralığında en iyi sonucu verir. Çünkü, "kapalı" veya "düşük tüketimli" tesislerin sayısı, "tam kapasite" çalışanlardan fazladır. Medyan, bu asimetriyi tam olarak yansıtmayabilir.

#### Kod İskeleti (Pseudo-Python)

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 1. Cold tesisleri belirle
cold_mask = df['has_history'] == False
cold_df = df[cold_mask].copy()
warm_df = df[~cold_mask].copy()

# 2. Normalizasyon
cold_df['norm_consumption'] = cold_df['consumption'] / cold_df['max_capacity']
warm_df['norm_consumption'] = warm_df['consumption'] / warm_df['max_capacity']

# 3. Clusterlama (Sadece Warm tesislerle eğit, Cold'a uygula)
features = ['log_capacity', 'sector_code', 'region_code']
# Warm tesislerin özelliklerini al
warm_features = warm_df[features].dropna()
kmeans = KMeans(n_clusters=15, random_state=42)
warm_features['cluster'] = kmeans.fit_predict(warm_features)

# Cold tesislere cluster ata
cold_features = cold_df[features].dropna()
cold_features['cluster'] = kmeans.predict(cold_features)

# 4. Cluster Bazlı Medyan Hesapla (Saat Bazlı)
# Her cluster ve her saat için Warm tesislerin medyan norm_consumption'ını al
cluster_hour_median = warm_df.groupby(['cluster', 'hour'])['norm_consumption'].median().unstack()

# 5. Cold Tahmini
cold_df['cluster'] = cold_features['cluster'].values
cold_df['hour'] = cold_df['hour']

# Cluster ve saate göre medyanı al
cold_df['predicted_norm'] = cold_df.apply(lambda row: cluster_hour_median.loc[row['cluster'], row['hour']], axis=1)

# 6. Gerçek Tüketime Dönüştür
cold_df['predicted_consumption'] = cold_df['predicted_norm'] * cold_df['max_capacity']

# 7. Shrinkage (Beta) Uygula
beta = 0.9 # Grid search ile optimize et
cold_df['final_prediction'] = cold_df['predicted_consumption'] * beta

# 8. GBDT Tahminlerini Birleştir
# Warm tesisler için GBDT tahmini, Cold tesisler için bu Bayes tahmini
final_predictions = np.where(cold_mask, cold_df['final_prediction'], gbm_predictions)
```

### Özet ve Beklenen Etki

*   **Mevcut Durum:** GBDT, Cold tesisler için "genel ortalama" tahmin ediyor. Bu, küçük tesislerde büyük log-hatalar yaratıyor.
*   **Yeni Durum:** Cold tesisler, kendi "kardeş" cluster'larının medyan davranışını, kapasitelerine orantılı olarak ve saat bazında tahmin ediyor.
*   **Beklenen Sonuç:**
    *   Cold Segment RMSLE: 1.82833 -> **1.40 - 1.50** aralığına düşmesi bekleniyor.
    *   Toplam Fold A RMSLE: 0.93873 -> **0.91 - 0.92** aralığına düşmesi bekleniyor.
    *   Test Setinde: Cold satırların %22.16 olması, bu iyileşmenin toplam skora **~0.01 - 0.015** puanlık doğrudan katkı sağlayacağı anlamına gelir.

Bu yaklaşım, "öğrenme" yerine "bilgi paylaşımı" ve "istatistiksel genelleme" üzerine kurulu olduğu için, veri eksikliği durumunda çok daha stabil ve güvenilir sonuçlar verir. GBDT'yi sadece Warm tesisler için kullanın, Cold tesisleri bu "cerrahi" müdahaleye bırakın.