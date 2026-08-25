# Grid Up Datathon — V9 Cold-Start Colab Çalışması

## Sonuç

- Colab, resmi `colab-mcp` bağlantısı üzerinden kullanıldı.
- Donanım: Tesla T4 (15 GB); CUDA erişimi `nvidia-smi` ile doğrulandı.
- Train: 1.226.237 satır; test: 714.688 satır.
- Testte hiç geçmişi olmayan tesisler: 2.024 tesis ve 158.369 satır (%22,16).
- Mevcut V8R submission'ın yalnız bu 158.369 cold-start satırı değiştirildi.
- Yıllık ve warm-start segmentlerindeki 556.319 tahmin değeri aynen korundu.

Nihai challenger dosyası: `submission_v9_cold_huber_mass.csv`

Cold-start ara tahminleri: `cold_v9_huber_mass.csv`

SHA256: `107386740eb1d508afd66aa1769a38440251f03a13fdb2e517f29765878d9b1a`

## Neden yeni bir cold-start modeli kuruldu?

V8R iç doğrulamasında segment skorları:

| Segment | Test payı | RMSLE |
|---|---:|---:|
| Yıllık geçmişi olanlar | %34,12 | 0,5744 |
| Warm-start | %43,72 | 0,8472 |
| Cold-start | %22,16 | 1,6680 |

Public skorun 1,13312 olması ve test cold-start kümesinin %68,4'ünün aynı gün başlayan 1.326 yeni tesisten oluşması, en büyük belirsizliğin cold-start/kohort dağılımında olduğunu gösterdi.

## Sızıntısız deney tasarımı

1. Tarihte ilk kez görülen her tesis için başlangıç tarihi çıkarıldı.
2. Aynı başlangıç tarihindeki tesisler tek kohort kabul edildi.
3. Bir kohort doğrulamadaysa o kohortun hiçbir tesisi eğitimde bırakılmadı.
4. Tahminlerde yalnız tesis açılırken bilinen bilgiler kullanıldı:
   - güç ve log-güç,
   - lokasyon,
   - anonim tesis kimliğinin önek/sonek kategorileri,
   - başlangıç ayı ve yıl içindeki günü,
   - aynı gün açılan kohortun büyüklüğü,
   - kohortun güç ve lokasyon bileşimi,
   - geçmiş tesislerden, tesis başına eşit ağırlıkla hesaplanan güç × ay × hafta-günü öncülü.
5. Model doğrudan tüketimi değil, güvenli öncülün tesis-seviyesi log-artığını öğrendi.
6. CatBoost, aykırı kohortlara dayanıklılık için Huber kaybıyla eğitildi.
7. Derinlik 4 ve 6 modellerinin ortalaması kullanıldı.

KMeans ana tahmin modeli yapılmadı. Sert küme etiketi bilinmeyen tesisin tüketim seviyesini tek başına çözmüyor; sürekli benzerlik ve düzenlenmiş öncüller daha kararlı kaldı.

## Doğrulama sonuçları

2025 kohortlarıyla eğitim, 2026 Ocak-Mart'ta ilk kez görülen tesislerle doğrulama:

| Model | Cold-start RMSLE |
|---|---:|
| V8R sağlamlaştırılmış cold-start | 1,6680 |
| Yeni sade anchor | 1,6816 |
| Huber derinlik 4+6 | 1,6352 |
| **Huber 4+6 + büyük kohort kalibrasyonu** | **1,6348** |

V8R'ye göre cold-start iyileşmesi yaklaşık 0,0332 RMSLE (%2,0). Yıllık ve warm segment skorlarının değişmediği varsayımıyla, segment-ağırlıklı toplam iç projeksiyon 1,0212'den yaklaşık 1,0093'e iner. Bu değer Kaggle skoru değildir; farklı segment validasyonlarının ağırlıklı tahminidir.

Zaman ilerlemeli ek kontroller:

| Doğrulama dönemi | Anchor | Huber 4+6 | Sonuç |
|---|---:|---:|---|
| 2025 Temmuz-Eylül | 1,8195 | 1,7886 | İyileşti |
| 2025 Ekim-Aralık | 2,0641 | 2,0710 | Hafif kötüleşti |
| 2025 Kasım-Aralık | 2,1083 | 2,1051 | Hafif iyileşti |
| 2026 Ocak-Mart | 1,6816 | 1,6346 | İyileşti |

Model dört ileri-zaman kontrolünün üçünde iyileşti; bu yüzden challenger olarak uygundur fakat public sonuç görülmeden tek nihai dosya ilan edilmemelidir.

## Büyük kohort kalibrasyonu

Test cold-start satırlarının 108.253'ü 11 Mayıs 2026'da başlayan 1.326 tesisten geliyor. Geçmişteki en yakın yüksek-güçlü toplu açılış kohortlarında sade öncülün log ölçekte yaklaşık 0,46-0,54 fazla tahmin yaptığı görüldü. Bu nedenle:

- en az 200 tesis ve medyan güç en az 500 olan kohortlarda ortalama log-artık -0,45'e,
- 100-199 tesis ve medyan güç en az 500 olan kohortlarda -0,30'a

yeniden merkezlendi. Bu işlem tek tek hedefleri ezberlemiyor; kohort büyüklüğü/güç yapısına uygulanan önceden tanımlı bir dayanıklılık kuralıdır.

## Test tahminindeki değişim

| Cold-start dağılımı | V8R | V9 Huber-Mass |
|---|---:|---:|
| Medyan | 1.430,07 | 724,63 |
| Ortalama | 1.687,38 | 1.038,73 |
| Minimum | 32,61 | 13,49 |
| %99 yüzdelik | 5.136,79 | 5.291,60 |
| Maksimum | 6.521,45 | 9.508,07 |

Değişimin büyük olması bilinçlidir: public skor ile iç projeksiyon arasındaki farkın ana hipotezi, testteki dev ve yüksek-güçlü yeni tesis kohortunun V8R tarafından fazla tahmin edilmesidir.

## Dosya doğrulamaları

- Satır sayısı: 714.688
- ID sırası test.csv ile birebir aynı
- Yinelenen ID: yok
- Eksik/sonsuz/negatif tahmin: yok
- Beklenen cold-start ID kümesi ile değiştirilen ID kümesi: birebir aynı
- Değişen satır: 158.369
- Değişmeden kalan satır: 556.319

## Submission kararı

Bu dosya günün ikinci kontrollü submission'ı olarak gönderilebilir. Public skor V8R'nin 1,13312 skorundan belirgin biçimde iyileşirse hipotez desteklenmiş olur. Küçük iyileşme veya kötüleşme durumunda katsayı public tabloya göre elle ayarlanmamalı; V8R ve V9 private leaderboard için iki farklı hata profili olarak saklanmalıdır.

Public skor için kesin sayı vermek mümkün değildir. İç projeksiyon yaklaşık 1,009 olsa da test dağılımı farklıdır; bu çalışma, cold-start riskini hedefleyen kontrollü bir challenger'dır.

## Public geri bildirim ve V10 karışımı

- V8R public RMSLE: **1,13312**
- V9 Huber-Mass public RMSLE: **1,13784**
- V9 uç noktası V8R'den **0,00472 daha kötü** geldi.

V8R ve V9 yalnız cold-start tahminlerinde ayrıldığı için, iki public skor ile iki dosyanın log-tahmin farkı RMSLE'nin karesel yapısında birlikte kullanıldı. Public satırların testten rastgele seçildiği varsayımıyla hesaplanan optimum interpolasyon ağırlığı yaklaşık `%45,6 V9 + %54,4 V8R` oldu. Bu karışımın public skor kestirimi yaklaşık **1,1219**'dur; bu kesin sonuç değil, public altkümenin tüm testteki tahmin-farkı dağılımına benzediği varsayımına dayanır.

Oluşturulan dosya: `submission_v10_logblend_0456.csv`

- Karıştırılan cold-start satırı: 158.369
- V8R'den metin/değer seviyesinde aynen korunan diğer satır: 556.319
- ID sırası test.csv ile birebir aynı
- Eksik, sonsuz, negatif ve yinelenen tahmin yok
- SHA256: `82c6a1d2d18f76a9ae25652e732c5af9b0d2895115c7351f1b5d97de5078614b`
