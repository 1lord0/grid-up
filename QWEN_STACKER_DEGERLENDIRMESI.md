# Grid-Up Datathon — Qwen 3.8-27B 3-Way Stacker ve AutoGluon Değerlendirmesi

**RAPOR: V20 MİMARİSİ KRİTİK DEĞERLENDİRMESİ**

**Yazar:** Kaggle Grandmaster / Tabular Veri Mimarı
**Konu:** "V20 — 3-Way GBDT Stacker + V8R Steel Anchor" Mimarisi Analizi
**Durum:** Eleştirel İnceleme ve Teknik Doğrulama

---

### 1. GENEL DEĞERLENDİRME: MANTIKLI VE SAĞLAM MI?

**Kısa Cevap:** Evet, bu mimari Kaggle Grandmaster standartlarına göre **mantıklı, sağlam ve istatistiksel olarak üstün** bir yaklaşımdır. Ancak, "sağlam" olması, implementasyon detaylarına (özellikle zaman sızıntısı ve bellek yönetimi) bağlıdır.

**Neden AutoGluon Değil? (Doğrulama)**
*   **Zaman Sızıntısı (Temporal Leakage):** AutoGluon'un varsayılan `TabularPredictor`'ı, veri seti zaman serisi doğasına sahipse (örneğin, tesis bazlı günlük/aylık üretim verileri), standart `K-Fold` yerine `TimeSeriesSplit` veya özel bir `groupby` stratejisi gerektirir. Varsayılan ayarlar, gelecekteki verinin geçmişe sızmasına neden olur. 1.7M satırda bu, CV skorunu yapay olarak şişirir (overfitting to future).
*   **Cold Start (2024 Tesisleri):** AutoGluon, hiperparametre optimizasyonunu (HPO) tüm veri seti üzerinde yapar. 2024'teki yeni tesisler için geçmiş veri yoksa, model bu tesisleri "gürültü" olarak algılayabilir veya onlara özel ağırlık veremez. Özel mimari, bu tesisler için `past_90d` gibi geriye dönük (lookback) özellikleri manuel olarak kontrol edebilir.
*   **Bellek Tüketimi:** AutoGluon, çoklu model (LightGBM, CatBoost, XGBoost, Neural Nets, etc.) ve çoklu fold üzerinde çalışır. 1.7M satır x ~50-100 özellik için, AutoGluon'un RAM tüketimi 32GB+ olabilir ve disk I/O darboğazı yaratabilir. Özel mimari, sadece 3 GBDT motoru ve kontrollü fold ile daha verimli çalışır.

**Sonuç:** AutoGluon, "hızlı başlangıç" için idealdir, ancak **zaman serisi + cold start + yüksek hacim** senaryosunda, manuel kontrol gerektiren özel bir mimari (V20) daha güvenilir ve optimize edilebilir.

---

### 2. OTOGLUON'A KIYASLA SOMUT AVANTAJLAR VE RİSKLER

#### A. Somut Avantajlar

| Özellik | AutoGluon (Varsayılan) | V20 Özel Mimari | Avantaj |
| :--- | :--- | :--- | :--- |
| **Fold Stratejisi** | Random K-Fold (Varsayılan) | Zaman Bazlı / Tesis Bazlı Grouped K-Fold | **Zaman sızıntısını tamamen ortadan kaldırır.** CV skoru gerçek test performansını yansıtır. |
| **Cold Start Yönetimi** | Genel model, yeni tesisleri genelleştirir | `past_90d` ve `max(0.05 * guc)` gibi manuel kurallar | **2024 tesisleri için daha düşük hata.** Model, yeni tesislerde "tahmin etme" yerine "güvenli taban" kullanır. |
| **Hiperparametre Kontrolü** | Otomatik (Optuna/Random Search) | Manuel + Optuna (Sadece Level-2) | **Daha az gürültü.** Sadece meta-öğrenici ağırlıkları optimize edilir, alt modellerin hiperparametreleri sabitlenir (deterministik). |
| **Bellek Verimliliği** | Çoklu model + çoklu fold | 3 GBDT + OOF matrisi | **%40-60 daha az RAM kullanımı.** Daha hızlı iterasyon. |
| **Özellik Mühendisliği** | Sınırlı (Otomatik) | Tam Kontrol (Fourier, Trend, Lokasyon) | **Domain bilgisi entegre edilir.** Fourier harmonikleri, mevsimselliği daha iyi yakalar. |

#### B. Somut Riskler

| Risk | Açıklama | Mitigasyon (Çözüm) |
| :--- | :--- | :--- |
| **Overfitting to OOF** | Level-2 meta-öğrenici, OOF skorlarına aşırı uyum sağlayabilir. | **Kısıtlı Optimizasyon:** Ağırlıkların toplamı 1 olmalı, negatif ağırlık yasak. Optuna'da `RMSLE` yerine `CV-RMSLE` (çoklu fold ortalaması) minimize edilmeli. |
| **Bellek Taşması (OOM)** | 1.7M satır x 3 model x OOF matrisi büyük olabilir. | **Chunking:** Veriyi tesis bazlı parçalara bölerek OOF hesapla. OOF matrisini `float32` olarak sakla. |
| **Model Korelasyonu** | 3 GBDT motoru benzer özellikler kullanırsa, stacker'ın faydası azalır. | **Özellik Ayrımı:** LGBM'e Fourier, CatBoost'e kategorik etkileşim, XGBoost'e regülarize edilmiş özellikler ver. |
| **V8R Anchor Hatası** | `1.13312` sabiti, test setinde değişen koşullara uyum sağlayamaz. | **Dinamik Anchor:** V8R'ı sabit değil, `train` setindeki ortalama hata oranına göre normalize et. |

---

### 3. 3 MOTORUN (LGBM + CatBoost + XGBoost) HARMANLANMASINDA EN KRİTİK TEKNİK KURAL

**Kural: "Özellik Ayrımı ve Korelasyon Kontrolü" (Feature Disentanglement & Correlation Control)**

**Neden Kritik?**
GBDT modelleri, benzer özellikler ve benzer hiperparametrelerle eğitildiğinde, yüksek korelasyonlu tahminler üretir. Bu durumda, Level-2 meta-öğrenici (stacker) sadece ağırlıkları ayarlayarak hata azaltamaz; çünkü modellerin hata yapıldığı noktalar aynıdır. **Stacker'ın faydası, modellerin "farklı hata profilleri" (error profiles) üretmesinden gelir.**

**Uygulama Adımları:**

1.  **Özellik Seti Ayrımı:**
    *   **LightGBM:** Zaman serisi özellikleri (Fourier harmonikleri, trend kalıntıları, lag özellikleri). LGBM, bu özelliklerde hızlı ve verimlidir.
    *   **CatBoost:** Kategorik özellikler (`ilce`, `bolge`, `guc_kategorisi`) ve bunların etkileşimleri. CatBoost, kategorik özellikleri native olarak işler ve overfitting'e karşı daha dirençlidir.
    *   **XGBoost:** Regülarize edilmiş özellikler (L1/L2 normları yüksek), anomali uç değerleri (outliers) ve düşük önemli özellikler. XGBoost, `reg_alpha` ve `reg_lambda` ile aşırı uyumu frenler.

2.  **Hiperparametre Farklılaştırması:**
    *   **LGBM:** `num_leaves` yüksek (50-100), `learning_rate` orta (0.05-0.1). Hızlı öğrenme.
    *   **CatBoost:** `depth` düşük (4-6), `l2_leaf_reg` yüksek (3-10). Düzgünleştirme.
    *   **XGBoost:** `max_depth` orta (6-8), `subsample` ve `colsample_bytree` düşük (0.7-0.8). Çeşitlilik.

3.  **Korelasyon Kontrolü:**
    *   OOF tahminleri arasında Pearson korelasyonunu hesapla.
    *   **Hedef:** Korelasyon < 0.85.
    *   Eğer korelasyon > 0.9 ise, modeller aynı şeyi öğreniyor demektir. Bu durumda, bir modeli değiştir veya özellik setini daha agresif olarak ayır.

**Matematiksel Doğrulama:**
Stacker'ın hata azaltma kapasitesi, modellerin kovaryans matrisine bağlıdır:
$$ \text{Var}(W_1 M_1 + W_2 M_2 + W_3 M_3) = \sum W_i^2 \text{Var}(M_i) + 2 \sum_{i<j} W_i W_j \text{Cov}(M_i, M_j) $$
Korelasyon (Cov) düşükse, toplam varyans (hata) daha hızlı azalır. Yüksek korelasyon, stacker'ın faydasını sıfıra yaklaştırır.

---

### SONUÇ VE ÖNERİLER

1.  **V20 Mimarisi Kabul Edilir.** AutoGluon'a kıyasla daha kontrollü, daha verimli ve domain bilgisiyle entegre edilmiş bir yapıdır.
2.  **Kritik Uyarı:** Zaman sızıntısını önlemek için **Grouped K-Fold** (tesis bazlı) veya **TimeSeriesSplit** (zaman bazlı) kullanmalısın. Random K-Fold **kesinlikle yasak**.
3.  **V8R Anchor:** `1.13312` sabitini, train setindeki ortalama RMSLE'ye göre normalize et. Test setinde koşullar değişirse, sabit anchor hata kaynağı olabilir.
4.  **Optuna:** Sadece Level-2 ağırlıklarını optimize et. Alt modellerin hiperparametrelerini sabit tut (deterministik). Bu, overfitting riskini azaltır.
5.  **Bellek:** OOF matrisini `float32` olarak sakla. 1.7M satır x 3 model = 5.1M değer. `float64` yerine `float32` kullanmak RAM'i yarıya indirir.

**Son Söz:** Bu mimari, Kaggle Grandmaster seviyesinde bir yaklaşım. Ancak, başarısı **implementasyon detaylarına** (fold stratejisi, özellik ayrımı, korelasyon kontrolü) bağlıdır. Bu detayları ihmal edersen, AutoGluon'dan farkın kalmaz.