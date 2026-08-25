# Grid-Up Datathon — Qwen 3.8-27B Model Incelemesi ve Gelistirme Raporu

Merhaba. Bir Kaggle Grandmaster ve Zaman Serisi Mimarı olarak, "Grid-Up Datathon" projesinin mevcut durumunu ve V14 mimarisini inceledim. Mevcut segmentasyon stratejiniz (Yıllık, Warm-Start, Cold-Start) çok sağlam bir temel oluşturuyor. Özellikle Cold-Start segmentindeki %22'lik pay, toplam RMSLE'yi belirleyen en kritik "kuyruk" (tail) riski. Mevcut RMSLE değerleriniz (özellikle Cold-Start'taki 1.668) bu segmentin hâlâ büyük iyileştirme potansiyeli taşıdığını gösteriyor.

Aşağıda, mevcut mimarinizi derinlemesine analiz eden ve RMSLE'yi düşürmek için somut, uygulanabilir 6 ileri düzey geliştirme önerisi sunuyorum.

### 1. Cold-Start Segmenti: Matris Ayristirma ve Embedding Tabanlı Yaklaşım
Mevcut durumda Cold-Start tesisler için Ridge regresyonu ve Bayes öncülleri kullanıyorsunuz. Ancak, tesislerin "kapasite" ve "lokasyon" özellikleri arasındaki non-lineer etkileşimleri yakalamak için daha güçlü bir temsil (representation) gerekir.

*   **Öneri:** **NMF (Non-Negative Matrix Factorization) veya SVD Tabanlı Tesis Embeddingleri.**
    *   **Mantık:** Tüketim verisi bir matris olarak düşünülebilir: $M_{t, d}$ (Tesis $t$, Gün $d$). Cold-Start tesisler için bu matrisin satırları eksiktir. Ancak, benzer kapasite ve lokasyona sahip *Warm-Start* veya *Yıllık* tesislerin tüketim profilleri vardır.
    *   **Uygulama:**
        1.  Tüm tesisler için (geçmişi olanlar) günlük tüketim vektörlerini alın.
        2.  Bu matrise NMF uygulayarak iki matris elde edin: $W$ (Tesis faktörleri) ve $H$ (Zaman/Mevsim faktörleri).
        3.  Cold-Start tesisler için $W$ matrisindeki satırı doğrudan hesaplayamazsınız. Bunun yerine, Cold-Start tesisin *statik özellikleri* (Kapasite, İl, İlçe, Tesis Tipi) ile *Warm-Start* tesislerin $W$ vektörleri arasında bir **Metrik Öğrenme (Metric Learning)** veya basit bir **K-NN (K-En Yakın Komşu)** eşleştirmesi yapın.
        4.  Cold-Start tesisin tahmini tüketimi, en yakın komşu tesislerin $H$ vektörlerinin ağırlıklı ortalaması ile çarpılarak elde edilir.
    *   **Neden İşe Yarar?** Bu yöntem, "benzer tesisler benzer tüketim profilleri gösterir" varsayımını matematiksel olarak kodlar. Ridge regresyonundan daha iyi genelleme sağlar çünkü lokasyon ve kapasitenin mevsimsel profil üzerindeki etkisini doğrudan öğrenir.

### 2. Hedef Dağılımı Kalibrasyonu: RMSLE'ye Özel Post-Processing
RMSLE, log-uzaydaki hata karesi ortalamasının kareköküdür. Standart GBDT modelleri (LightGBM/CatBoost) genellikle MSE veya MAE ile optimize edilir. Bu, log-uzayda simetrik olmayan hatalara (özellikle küçük değerlerdeki büyük nispi hatalara) karşı hassasiyet kaybına yol açar.

*   **Öneri:** **Log-Uzayda Hata Minimizasyonu ve Asimetrik Kesikli Kalibrasyon (Asymmetric Quantile Calibration).**
    *   **Mantık:** RMSLE, $\sqrt{\frac{1}{n}\sum (\ln(y_i) - \ln(\hat{y}_i))^2}$ olarak tanımlanır. Bu, log-uzayda MSE'ye eşdeğerdir. Ancak, gerçek dünya verilerinde tüketim genellikle sağa çarpık (right-skewed) dağılır.
    *   **Uygulama:**
        1.  Modelinizi doğrudan `log1p(y)` hedefine karşı eğitin (Zaten yapıyor olabilirsiniz, ancak kontrol edin).
        2.  **Kritik Adım:** Tahmin çıktısını `expm1` ile geri dönüştürmeden önce, log-uzayda bir **Bayes Düzeltmesi** uygulayın.
        3.  Doğrulama setinde, modelin log-uzaydaki ortalama hatasını ($\bar{e}_{log}$) hesaplayın. Eğer model sistematik olarak düşük tahmin ediyorsa (negatif bias), log-uzaydaki tahminlere bu bias'ı ekleyin.
        4.  Daha ileri düzey: **Quantile Regression** kullanın. RMSLE, ortalama hatayı minimize eder. Ancak, soğuk başlangıç tesislerinde yüksek varyans vardır. `quantile=0.5` yerine, doğrulama setinde RMSLE'yi minimize eden optimal quantile'ı (örneğin 0.45 veya 0.55) arayın. Bazen ortalama (mean) tahmini, RMSLE için en iyi tahmin değildir, çünkü log-uzayda hata dağılımı simetrik olmayabilir.

### 3. Cross-Validation Stratejisi ve Sızıntı (Leakage) Önleme
Zaman serilerinde standart K-Fold CV kullanmak, gelecekteki bilginin geçmişe sızmasına (leakage) neden olur. Özellikle "Yıllık Gecikmeli" segmentte bu risk çok yüksektir.

*   **Öneri:** **Time-Series Split ile Birleşik "Purged" ve "Embargo" CV.**
    *   **Mantık:** Lag özellikleri (364 gün önceki değer) kullanıldığında, test setindeki bir günün lag'i, eğitim setindeki bir güne denk gelebilir. Bu, modelin "cevabı" eğitim sırasında görmesine neden olur.
    *   **Uygulama:**
        1.  **Purging:** Eğitim ve test setleri arasında, lag derinliğine (365 gün) eşit bir "boşluk" (gap) bırakın.
        2.  **Embargo:** Test setinin hemen öncesindeki verileri eğitimden tamamen çıkarın.
        3.  **Segment Bazlı CV:** Her segment için ayrı CV stratejisi uygulayın.
            *   *Yıllık:* Yıl bazlı split (Örn: 2021-2022 eğitim, 2023 test).
            *   *Warm-Start:* Son 60 gün eğitim, sonraki 30 gün test.
            *   *Cold-Start:* Tesis bazlı split (Bazı tesisler eğitimde, bazıları testte). Bu, modelin yeni tesislere genelleme gücünü test eder.
    *   **Kritik Kontrol:** `sklearn.model_selection.TimeSeriesSplit` kullanın, ancak lag'ler için özel bir `PurgedKFold` implementasyonu yazmanız gerekebilir.

### 4. Feature Engineering: Dinamik Lokasyon Hiyerarşisi ve İklim Etkileşimleri
Mevcut lokasyon hiyerarşisi (il, ilçe) statik. Ancak, elektrik tüketimi iklim koşullarına (sıcaklık, nem) ve tatil takvimine dinamik olarak tepki verir.

*   **Öneri:** **Hedef Kodlama (Target Encoding) ile Dinamik Lokasyon ve İklim Etkileşim Özellikleri.**
    *   **Mantık:** "İstanbul" ve "Ankara" aynı kapasitede tesisler için farklı tüketim profilleri gösterir. Bu fark, iklim ve kültürel alışkanlıklardan kaynaklanır.
    *   **Uygulama:**
        1.  **Dinamik Target Encoding:** Her lokasyon (il/ilçe) için, *geçmişteki* ortalama log-tüketimi hesaplayın. Ancak, bu değeri hesaplarırken **sızıntıyı önlemek için** sadece geçmiş verileri kullanın (expanding window).
        2.  **İklim Etkileşimi:** Meteoroloji verisi varsa (veya yoksa, il bazlı ortalama sıcaklık), `Sıcaklık * Kapasite` ve `Sıcaklık * Mevsim` özellikleri ekleyin.
        3.  **Tatil Etkisi:** Resmi tatiller, hafta sonları ve yerel tatiller için binary özellikler ekleyin. Ayrıca, "Tatilden sonraki ilk iş günü" gibi faz özellikleri (phase features) ekleyin.
        4.  **Kapasite Bazlı Oranlar:** `Tüketim / Kapasite` oranını log-uzayda modelleyin. Bu, tesislerin büyüklük farkını normalize eder ve modelin "verimlilik" veya "doluluk" oranını öğrenmesini sağlar.

### 5. Warm-Start Segmenti: Hibrit Zaman Serisi Modeli (Prophet/ARIMA + GBDT)
Mevcut durumda Warm-Start için CatBoost/LightGBM kullanıyorsunuz. Ancak, kısa vadeli (30-60 gün) tahminlerde, mevsimsellik ve trendi yakalamak için klasik zaman serisi modelleri GBDT'den daha iyi olabilir.

*   **Öneri:** **Stacking ile GBDT ve Zaman Serisi Modeli Entegrasyonu.**
    *   **Mantık:** GBDT, non-lineer etkileşimleri iyi yakalar ama uzun vadeli mevsimselliği zayıf modelleyebilir. Prophet veya ARIMA, mevsimselliği ve trendi iyi yakalar ama non-lineer özelliklerden (kapasite, lokasyon) faydalanamaz.
    *   **Uygulama:**
        1.  Her Warm-Start tesisi için, son 60 günlük veriyi kullanarak basit bir **Prophet** veya **Holt-Winters** modeli eğitin. Bu model, tesisin kendi trendini ve haftalık mevsimselliğini yakalar.
        2.  Bu modelin tahminlerini, GBDT modelinin girdisi olarak kullanın (`prophet_pred` feature).
        3.  GBDT modeli, `prophet_pred`, `kapasite`, `lokasyon`, `iklim` gibi özelliklerle birlikte eğitilir.
        4.  Sonuç: GBDT, Prophet'in mevsimsel tahminini "düzelten" bir residual model haline gelir. Bu, özellikle trend değişimi olan tesislerde RMSLE'yi düşürür.

### 6. V14 Mimarisindeki "Soft Clustering" ve "Risk-Hedged Blend" İyileştirmesi
Mevcut V14'te K-Means tabanlı 3 arketip ve yumuşak olasılıklar kullanıyorsunuz. Bu iyi bir başlangıç, ancak K-Means, tüketim verisinin non-lineer yapısını ve lokasyon etkisini tam olarak yakalayamayabilir.

*   **Öneri:** **Gaussian Mixture Model (GMM) ile Soft Clustering ve Dinamik Ağırlıklandırma.**
    *   **Mantık:** K-Means, sert sınırlar çizer. GMM ise, her tesisin her kümeye ait olma olasılığını (soft assignment) verir. Ayrıca, GMM, kümelerin varyansını da modelleyerek, "yüksek belirsizlik" olan tesisleri daha iyi ayırt eder.
    *   **Uygulama:**
        1.  Tüketim profillerini (log-uzayda) GMM ile kümeleyin.
        2.  Her tesis için, her kümeye ait olma olasılığını ($P(k|t)$) hesaplayın.
        3.  Tahmin yaparken, her küme için ayrı bir model (veya ortalama tüketim profili) kullanın.
        4.  Son tahmin, $\sum_{k} P(k|t) \cdot \hat{y}_k$ şeklinde hesaplanır.
        5.  **Risk-Hedged Blend:** Mevcut 0.75/0.25 blend yerine, **doğrulama setindeki RMSLE'ye göre dinamik ağırlıklandırma** yapın. Her segment için (Yıllık, Warm, Cold) ayrı ağırlıklar optimize edin. Örneğin, Cold-Start'ta V14'ün ağırlığı 0.6, V8R'ın ağırlığı 0.4 olabilir. Bu ağırlıkları, doğrulama setinde RMSLE'yi minimize eden değerler olarak arayın.

### Özet ve Öncelik Sıralaması

| Öncelik | Görev | Beklenen Etki | Uygulama Zorluğu |
| :--- | :--- | :--- | :--- |
| **1** | **Cold-Start: NMF/Embedding** | Yüksek (RMSLE 1.668 -> 1