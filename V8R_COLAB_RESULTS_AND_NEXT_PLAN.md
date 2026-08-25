# Grid Up Datathon — V8R Colab Sonuçları ve Devam Planı

## Durum

- Colab bağlantısı resmi `colab-mcp` sunucusu üzerinden kuruldu.
- Donanım `Tesla T4 (15 GB)` ve CUDA erişimi `nvidia-smi` ile doğrulandı.
- Train/test dosyaları Colab'a aktarıldı; deneyler gerçek veri üzerinde çalıştırıldı.
- Nihai yerel aday: `submission_v8r_verified_final.csv`
- Satır sayısı: 714.688
- Sample submission ile ID sırası: birebir aynı
- Eksik, sonsuz, negatif veya yinelenen ID: yok
- SHA256: `271FCA5168D3AE7AA86EABD25736C2E6BE9A0BF76E337A2A07D76ABEA13E2A37`

## Kullanılan sızıntısız yapı

Test satırları üç ayrı problem olarak ele alındı:

1. **Yıllık geçmişi olanlar — 243.839 satır (%34,12)**
   - 364/365/366/371 günlük gecikmeler
   - geçmişten öğrenilen daraltılmış yıllık büyüme
   - tesis seviyesi + emsal mevsimsellik aktarımı
   - doğrulamada en iyi karışım: %30 yıllık-gecikme, %70 mevsimsellik aktarımı

2. **Geçmişi var fakat yıllık eşleşmesi olmayanlar — 312.480 satır (%43,72)**
   - tesisin son 30 günlük seviyesini emsal mevsim profiline taşıyan warm-start model

3. **Hiç geçmişi olmayan tesisler — 158.369 satır (%22,16)**
   - tesislerin örnek sayısına göre değil, tesis başına eşit oyla hesaplanan güç/ay/hafta-günü öncülleri
   - lokasyon, güç, kimlik öneki ve takvim kırılımlarının Ridge ile düzenlemeli birleşimi
   - aşırı uyuma karşı Ridge tahmini %75, sade güç-tipi/güç/emsal çıpası %25

## Colab ve yerel doğrulama sonuçları

| Segment / model | RMSLE | Doğrulama satırı |
|---|---:|---:|
| Yıllık gecikme ham | 1,2546 | 53.797 |
| Yıllık büyüme düzeltmeli | 0,7375 | 53.797 |
| Mevsimsellik aktarımı | 0,6134 | 53.797 |
| **Yıllık segment kazanan karışım** | **0,5744** | **53.797** |
| **Warm-start** | **0,8472** | **156.114** |
| Cold-start sade peer | 1,7338 | 26.707 |
| Cold-start Ridge | 1,6742 | 26.707 |
| **Cold-start sağlamlaştırılmış karışım** | **1,6680** | **26.707** |
| Doğrudan CatBoost cold-start | 2,6442 | 26.707 |

Test segment oranlarıyla hesaplanan yaklaşık toplam RMSLE **1,0212**'dir. Bu sayı farklı segment doğrulamalarının ağırlıklı projeksiyonudur; Kaggle'ın gizli test skoru veya tek bir birleşik fold skoru değildir.

## Alınan kararlar

- Doğrudan CatBoost cold-start modeli, T4 üzerinde 1.200 iterasyon eğitilmesine rağmen gerçek yeni tesislerde 2,6442 verdiği için elendi.
- Yaş/ramp-up düzeltmesi cold-start skorunu 1,7338'den 1,7434'e kötüleştirdiği için elendi.
- KMeans etiketi ana tahmin olarak kullanılmadı. Sert küme etiketi, benzerlik bilgisini kaybediyor ve hiç görülmeyen tesis problemini tek başına çözmüyor.
- Sadece günün yılı sinüsünden üretilen “hava” değişkeni gerçek hava verisi kabul edilmedi.
- Önceki `accuracy_first` skoru, kesim tarihinden sonraki özetler ve aynı-fold yönlendirici riski yüzünden güvenilir referans kabul edilmedi.

## En yüksek doğruluk için sıradaki deneyler

### 1. Resmi geri bildirim

`submission_v8r_verified_final.csv` tek bir kontrollü public submission olarak gönderilmeli. Public skor, yalnızca model sıralaması için kullanılmalı; %30 public parçaya elle aşırı uyum yapılmamalı.

### 2. Warm-start residual CatBoost

- Hedef doğrudan tüketim değil, mevcut warm-start tahmininin log-artığı olmalı.
- Eğitim kesiti 2025 sonu, doğrulama 2026 Ocak-Mart olmalı.
- Model ancak toplam warm RMSLE'yi ve her ayın skorunu birlikte iyileştirirse eklenmeli.
- Kabul eşiği: en az 0,01 RMSLE ve hiçbir ayda belirgin bozulma olmaması.

### 3. Yıllık segmentte satır-bazlı yönlendirme

- Sabit %30/%70 yerine gecikme sayısı, gecikmeler arası yayılım, son dönem oynaklığı ve tesis geçmiş uzunluğuna göre ağırlık öğrenilmeli.
- Yönlendirici yalnızca tarih-dışı OOF tahminlerden eğitilmeli.

### 4. Cold-start düşük-rank mevsim profili

- Tesis × (ay, hafta-günü) matrisi SVD/NMF ile 4-8 sürekli bileşene ayrılmalı.
- Bileşenler güç, lokasyon ve güvenli kimlik öneklerinden tahmin edilmeli.
- Sonuç mevcut Ridge/anchor modeline eklenecek ayrı bir challenger olmalı; tek başına ana model yapılmamalı.

### 5. Kohort dayanıklılığı

- Testteki büyük 11 Mayıs 2026 başlangıç kohortu nedeniyle, yeni tesis backtesti başlangıç tarihini grup kabul ederek yapılmaya devam edilmeli.
- Kohort bazında çok oynak modeller, ortalama skor iyi görünse bile elenmeli.

### 6. Gerçek dış veri — yalnızca yazılı izin sonrası

- İzin gelirse lokasyon/gün bazlı sıcaklık, soğutma derece-günü, nem ve resmi tatil değişkenleri eklenebilir.
- Kaynak ve indirme tarihi notebook'ta açıkça belgelenmeli.
- İzin gelmeden nüfus, gelir, fabrika bölgesi veya gerçek hava verisi eklenmemeli.

## Submission stratejisi

- İlk kontrollü aday: `submission_v8r_verified_final.csv`
- İkinci aday yalnızca sızıntısız OOF'de V8R'yi geçen ve aylara/kohortlara dayanıklı bir model olmalı.
- Sadece public leaderboard yükseldi diye kör ensemble veya manuel katsayı eklenmemeli.
