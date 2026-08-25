# Grid-Up Datathon — Qwen 3.8-27B SOTA Makine Öğrenmesi ve Açık Kaynak Model Araştırma Raporu

Merhaba. Bir Zaman Serisi ve Tabular Makine Öğrenmesi Araştırmacısı olarak, karşılaştığınız problemi derinlemesine analiz ettim. Durumunuz, klasik "Cold Start" (Soğuk Başlangıç) probleminin, RMSLE gibi asimetrik bir metrikle birleştiği ve veri dağılımının (zero-inflation) ağır kuyruklu olduğu karmaşık bir senaryodur.

Mevcut durumunuzda Warm tarafında 0.825 RMSLE oldukça iyi bir seviyedir. Ancak Cold tarafında 1.79-1.80 bandında takılıp kalmanız, GBDT modellerinin (XGBoost/LightGBM) **lokal interpolasyon** doğası gereği, geçmiş verisi olmayan (sıfır geçmiş) örneklerde "ortalama"ya (mean) veya "medyan"a (median) çökme eğiliminde olduğunu gösterir. RMSLE metriğinde, gerçek değeri $y$ olan bir örnekte tahminin $\hat{y}$ olması durumunda hata $\log(\frac{\hat{y}+1}{y+1})$ olarak hesaplanır. Eğer $y$ küçükse ve siz 0 tahmin ederseniz hata $\log(1/(y+1))$ olur ki bu, $y$'nin küçük olduğu durumlarda (örneğin $y=1$ ise $\log(1/2) \approx -0.69$, karesi 0.48) ciddi bir ceza doğurur. Ancak asıl sorun, **aktif** bir tesise 0 basmanızdır. Bu durumda hata $\log(1/(y+1))$ olur. Eğer $y$ büyükse (örneğin 1000), hata $\log(1/1001) \approx -6.9$ olur, karesi ~47. Bu, tek bir hatanın toplam skoru patlatmasına neden olur.

Aşağıda, 2024-2026 literatürü ve Kaggle Grandmaster pratikleri ışığında, bu problemi çözmek için 4 ana eksende derinlemesine bir araştırma raporu sunuyorum.

---

# ARAŞTIRMA RAPORU: Elektrik Dağıtım Şebekesi Cold-Start Tahmini ve RMSLE Optimizasyonu

## 1. Zaman Serisi Foundation Modelleri (TSFM) ve Cold-Start Uygulanabilirliği

### 1.1. Literatür Özeti ve Model Karşılaştırması
2024-2025 döneminde zaman serisi alanında "Foundation Models" (Temel Modeller) devrimi yaşanmıştır. Bu modeller, milyarlarca zaman serisi üzerinde ön-eğitimli (pre-trained) olup, az miktarda veriyle (few-shot) veya sıfır örnekli (zero-shot) tahmin yapabilirler.

*   **Amazon Chronos (2024):** Tokenizer tabanlı bir dil modeli yaklaşımı kullanır. Zaman serilerini token'lara dönüştürür ve dil modeli mimarisini (Transformer) kullanır. **Avantajı:** Sıfır-örnekli performansı yüksektir. **Dezavantajı:** Uzun bağlam (long-context) desteği sınırlıdır ve elektrik tüketimi gibi yüksek frekanslı, mevsimsel deseni güçlü verilerde "noise" (gürültü) filtreleme konusunda GBDT kadar keskin değildir.
*   **Google TimesFM (2024):** 200M parametreli, patch-based transformer mimarisi. **Avantajı:** Çok uzun geçmiş verilerini (long history) işleyebilir. **Dezavantajı:** Cold-start için "prompt" olarak geçmiş verisi gerektirir. Sizin Cold setinizde geçmiş verisi **sıfır** olduğu için, TimesFM'i doğrudan Cold set için kullanmak imkansızdır. Ancak, **Warm setlerin geçmişini kullanarak Cold setlerin "prototip"lerini oluşturmak** için kullanılabilir.
*   **Salesforce MOIRAI (2024):** Çoklu zaman serisi (multi-series) ve çoklu frekans (multi-frequency) destekli. **Avantajı:** Farklı uzunluklardaki serileri aynı modelde işleyebilir. **Dezavantajı:** Hala geçmiş verisi gerektirir.
*   **Lag-Llama (2024):** Lag features'a odaklanır. **Dezavantajı:** Cold-start'ta lag feature'ları yoktur.

### 1.2. Strateji: "Transfer Learning via Prototyping" (Prototipleme ile Aktarım Öğrenmesi)

Cold set için doğrudan TSFM kullanmak yerine, **Warm setlerden öğrenilen desenleri Cold setlere aktaran hibrit bir mimari** öneriyorum.

**Mimari Tasarım:**
1.  **Warm Set Analizi:** 7.036 tesisin tamamı için, TimesFM veya Chronos kullanarak, her tesisin "karakteristik tüketim profili"ni (profile) çıkarın. Bu profil, tesisin trafo gücüne, lokasyonuna ve sektörüne (varsa) göre normalize edilmiş bir vektördür.
2.  **Cold Set Prototipleme:** Cold set için geçmiş verisi yoktur. Ancak `ilce`, `bolge` ve `trafo_gucu` bilgisi vardır.
    *   Aynı `ilce` ve benzer `trafo_gucu` aralığındaki **Warm tesislerin** TSFM çıktıları (veya GBDT çıktıları) üzerinden **ortalama bir "prototip seri"** oluşturun.
    *   Bu prototip seriyi, Cold tesisin "sanal geçmişi" olarak kullanın.
3.  **Fine-Tuning:** Bu sanal geçmişle, TSFM modelini Cold tesisler için hafifçe fine-tune edin (veya sadece inference yapın).

**Uygulanabilirlik Değerlendirmesi:**
*   **Zorluk:** TSFM'ler genellikle GPU gerektirir ve inference süresi uzundur. 158.369 Cold satır için bu maliyetli olabilir.
*   **Öneri:** TSFM'leri doğrudan final model olarak değil, **feature üretici** olarak kullanın. TSFM'den elde edilen "trend" ve "mevsimsellik" bileşenlerini, GBDT modeline ek feature olarak verin. Bu, GBDT'nin Cold setlerde "kör" kalmasını engeller.

---

## 2. Tabular Deep Learning & Hybrid Mimariler: Entity Embeddings ile Cold-Start Çözümü

GBDT modellerinin Cold setlerde başarısız olmasının temel nedeni, **kategorik değişkenlerin (lokasyon) etkisini öğrenememesidir**. GBDT, `ilce` gibi bir değişkeni sadece bir "split" noktası olarak görür. Ancak, aynı ilçedeki tesislerin tüketim davranışları benzerdir. Deep Learning, bu benzerliği **Embedding** vektörleri ile öğrenir.

### 2.1. Önerilen Mimari: "Hierarchical Entity-Enhanced GBDT-Deep Hybrid"

Bu mimari, GBDT'nin tabular verilerdeki güçlü yönlerini ve Deep Learning'in hiyerarşik lokasyon bilgisi öğrenme yeteneğini birleştirir.

**Mimari Diyagramı:**

```
[Input Features]
   |
   +---> [Numerical Features] (Trafo Gücü, Gün, Ay, Hava Durumu vb.)
   |
   +---> [Categorical Features] (Ilce, Bolge, Trafo Tipi)
           |
           +---> [Entity Embedding Layer]
                   |
                   +---> Ilce Embedding (Dim: 16)
                   +---> Bolge Embedding (Dim: 8)
                   +---> Trafo_Gucu Bucket Embedding (Dim: 4)
                   |
                   +---> [Concatenate]
                   |
                   +---> [Dense Layer (64 units, ReLU)]
                   |
                   +---> [Dropout (0.2)]
                   |
                   +---> [Output: Cold_Start_Prior] (Skalor)
```

**Adım Adım Uygulama:**

1.  **Entity Embeddings:**
    *   `ilce` ve `bolge` için öğrenilebilir embedding vektörleri tanımlayın.
    *   Bu embedding'ler, aynı ilçedeki tesislerin birbirine yakın vektörlerle temsil edilmesini sağlar.
    *   **Önemli:** Embedding'leri sadece Cold set için değil, **tüm veri seti (Warm + Cold)** üzerinde eğitin. Warm setlerdeki güçlü sinyaller, embedding'lerin daha iyi öğrenilmesini sağlar. Cold setlerdeki tesisler, aynı ilçedeki Warm tesislerin embedding'lerinden "yararlanır".

2.  **Hybrid Output:**
    *   GBDT modeli (LightGBM) ana tahmini yapar.
    *   Deep Learning modeli (yukarıdaki mimari), Cold set için bir **"öncelik skoru" (prior score)** üretir.
    *   Final tahmin: $\hat{y}_{final} = \hat{y}_{GBDT} \times \hat{y}_{DL}$ veya $\hat{y}_{final} = \hat{y}_{GBDT} + \alpha \cdot \hat{y}_{DL}$ (log-uzayda).
    *   **Neden?** GBDT, Cold setlerde ortalama değere çöker. DL modeli, lokasyon hiyerarşisi sayesinde, "Bu ilçedeki tesisler genellikle ortalamanın %20 üzerindedir" bilgisini öğrenir ve GBDT'nin tahminini düzeltir.

3.  **Alternatif: FT-Transformer (2024-2025 Trendi):**
    *   FT-Transformer, tabular veriler için Transformer mimarisini kullanır.
    *   Her feature'ı bir token olarak ele alır.
    *   **Avantajı:** Feature'lar arasındaki etkileşimleri (interaction) otomatik olarak öğrenir.
    *   **Uygulama:** FT-Transformer'ı, GBDT'nin yerine değil, **ensemble** olarak kullanın. FT-Transformer'ın çıktısı, GBDT'nin çıktısıyla birleştirilir.

---

## 3. İki Aşamalı Hurdle / Zero-Inflated RMSLE Modelleri

Sıfır çökmesi (zero-collapse) problemi, RMSLE metriğinin doğasından kaynaklanır. RMSLE, $\log(y+1)$ uzayında MSE'yi minimize eder. Eğer $y=0$ ise, $\log(1)=0$. Eğer $\hat{y}=0$ ise, hata 0. Ancak, $y>0$ ve $\hat{y}=0$ ise hata $\log(1/(y+1))$ olur. Bu, **asimetrik** bir cezadır.

### 3.1. Hurdle Model Mimarisi

Bu yaklaşım, problemi iki alt probleme böler:

1.  **Sınıflandırma (Classifier):** Tesisin o gün **aktif** mi (y > 0) yoksa **pasif** mi (y = 0)?
2.  **Regresyon (Regressor):** Tesis aktifse, tüketimi ne kadar?

**Formülasyon:**

*   **Adım 1: Aktiflik Sınıflandırması**
    *   Hedef: $I(y > 0)$ (Binary)
    *   Model: LightGBM Classifier (LogLoss metrik)
    *   Çıktı: $P_{active} = P(y > 0 | X)$

*   **Adım 2: Miktar Regresyonu**
    *   Hedef: $y$ (Sadece $y > 0$ olan örnekler üzerinde eğitilir)
    *   Model: LightGBM Regressor (RMSLE metrik)
    *   Çıktı: $\hat{y}_{reg}$

*   **Adım 3: Final Tahmin Birleştirme**
    *   Naif yaklaşım: $\hat{y}_{final} = P_{active} \times \hat{y}_{reg}$
    *   **Sorun:** Bu yaklaşım, $P_{active}$ küçükse tahmini çok düşük yapar. RMSLE'de bu, "under-prediction" cezasına yol açar.
    *   **Düzeltilmiş Yaklaşım (Log-Space Blending):**
        *   Log-uzayda çalışın: $z = \log(y+1)$
        *   Sınıflandırıcı, $P(z > 0)$'i tahmin eder.
        *   Regresör, $E[z | z > 0]$'i tahmin eder.
        *   Final log-tahmin: $\hat{z}_{final} = P_{active} \times \hat{z}_{reg}$
        *   Final tahmin: $\hat{y}_{final} = \exp(\hat{z}_{final}) - 1$

**Neden Bu İşe Yarar?**
*   Cold setlerde, tesislerin çoğu "aktif"tir (elektrik kesintisi yoksa). Sınıflandırıcı, lokasyon ve trafo gücüne dayanarak, "Bu tesisin aktif olma olasılığı %95'tir" der.
*   Regresör, "Aktifse, ortalama 100 kWh tüketir" der.
*   Final tahmin: $0.95 \times 100 = 95$. Bu, GBDT'nin 0'a çökmesinden çok daha iyidir.

**Uygulama İpuçları:**
*   Sınıflandırıcı için **feature'lar**: Lokasyon, trafo gücü, günün tipi (hafta sonu/hafta içi), mevsim.
*   Regresör için **feature'lar**: Sınıflandırıcının çıktısı ($P_{active}$) da bir feature olarak eklenmelidir.

---

## 4. Bayesyen ve Conformal Prediction Tabanlı Asimetrik Çıktı Kalibrasyonu

RMSLE metriği, $\log(y+1)$ uzayında MSE'ye eşdeğerdir. MSE, **ortalama** (mean) değerini minimize eder. Ancak, log-uzayda ortalama, orijinal uzayda **geometrik ortalama**'ya karşılık gelir.

### 4.1. Problem: "Mean vs. Median" Kayması

*   Eğer veri dağılımı sağa çarpık (right-skewed) ise (elektrik tüketimi genellikle öyledir), log-uzayda ortalama, orijinal uzayda medyanın **altında** kalır.
*   RMSLE, küçük tahminleri (under-prediction) büyük tahminlerden (over-prediction) daha az cezalandırır mı? Hayır, RMSLE simetrik değildir.
    *   $\log(\hat{y}/y)$ hata.
    *   $\hat{y} < y$ ise hata negatif, $\hat{y} > y$ ise pozitif.
    *   Kare alındığında, büyük sapmalar daha çok cezalandırılır.
    *   Ancak, **log-uzayda MSE minimize etmek**, orijinal uzayda **geometrik ortalama**'ya yakınsar.
    *   Elektrik tüketimi gibi sağa çarpık dağılımlarda, geometrik ortalama, aritmetik ortalamanın altındadır. Bu, modelin **sistemli olarak düşük tahmin** yapmasına neden olur.

### 4.2. Çözüm: Asimetrik Kalibrasyon (Asymmetric Calibration)

**Yöntem 1: Log-Space Offset (Basit ve Etkili)**

1.  Modelinizi log-uzayda eğitin: $z = \log(y+1)$
2.  Tahminleri alın: $\hat{z}$
3.  **Offset Hesaplama:**
    *   Validasyon setinde, $\hat{z}$ ve $z$ arasındaki farkı hesaplayın.
    *   $offset = \text{mean}(z - \hat{z})$
    *   Ancak, bu offset'i **Cold set için ayrı** hesaplayın.
    *   Cold setlerde, modelin sistematik olarak düşük tahmin yaptığı gözlemleniyorsa, offset pozitif olacaktır.
4.  **Final Tahmin:**
    *   $\hat{z}_{calibrated} = \hat{z} + offset_{cold}$
    *   $\hat{y}_{final} = \exp(\hat{z}_{calibrated}) - 1$

**Yöntem 2: Conformal Prediction ile Güven Aralığı (2024-2025 Trendi)**

Conformal Prediction, modelin belirsizliğini (uncertainty) ölçmek için kullanılır.

1.  **Nonconformity Score:** Her örnek için, $s_i = |z_i - \hat{z}_i|$
2.  **Quantile Hesaplama:** Validasyon setinde, $s_i$ değerlerinin $(1-\alpha)$ quantilini bulun.
3.  **Güven Aralığı:**
    *   $\hat{z}_{lower} = \hat{z} - q_{1-\alpha}$
    *   $\hat{z}_{upper} = \hat{z} + q_{1-\alpha}$
4.  **RMSLE Optimizasyonu:**
    *   RMSLE, asimetrik olduğu için, güven aralığının **alt sınırını** (lower bound) kullanmak, "under-prediction" riskini azaltır.
    *   Ancak, bu, "over-prediction" riskini artırır.
    *   **Optimal Nokta:** $\hat{z}_{final} = \hat{z} + \lambda \cdot q_{1-\alpha}$, burada $\lambda \in [0, 1]$ bir hiperparametredir.
    *   $\lambda$'yı validasyon setinde grid search ile optimize edin.

**Uygulama Önerisi:**
*   Cold setlerde, modelin belirsizliği yüksektir. Conformal Prediction, bu belirsizliği ölçer.
*   $\lambda$'yı, Cold set validasyonunda (varsa) veya Warm setlerin "cold-like" alt kümesinde optimize edin.
*   Bu, modelin "güvenmediği" durumlarda, tahmini yukarı doğru kaydırarak, sıfır çökmesini engeller.

---

## 5. Entegre Uygulama Planı (Action Plan)

Aşağıdaki adımları sırasıyla uygulayın:

### Adım 1: Veri Hazırlığı ve Feature Engineering
1.  **Lokasyon Hiyerarşisi:** `ilce` ve `bolge` için **Target Encoding** yerine, **Entity Embeddings** kullanın (Adım 2'de detaylandırıldı).
2.  **Trafo Gücü Bucket'ları:** Trafo gücünü sürekli bir feature olarak değil, **kategorik bucket'lara** (örn: 0-100, 100-500, 500-1000, >1000) ayırın. Bu, GBDT'nin ve DL'nin öğrenmesini kolaylaştırır.
3.  **Zaman Feature'ları:** Gün, Ay, Hafta İçi/Sonu, Mevsim, Tatil Günleri.
4.  **Hava Durumu (Varsa):** Sıcaklık, nem. Elektrik tüketimi ile güçlü korelasyon vardır.

### Adım 2: Model Eğitimi
1.  **Hurdle Model:**
    *   **Classifier