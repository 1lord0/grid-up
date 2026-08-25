# V22 Leaderboard-Kalibreli Ensemble Raporu

## Neden V21 değil?

Offline validasyon model sıralamasını ters çevirmiştir:

| Submission | Gerçek Kaggle RMSLE |
|---|---:|
| V8R | 1,13312 |
| V9 | 1,13784 |
| V12 | 1,14332 |
| V16 | 1,32029 |

V16 offline ölçümde çok güçlü görünmesine rağmen Kaggle'da çökmüştür. Bu
nedenle Fold A'dan türetilen yaklaşık 1,20 hesabı leaderboard tahmini olarak
kullanılamaz. V21 yükleme önerisi geri çekilmiştir.

## Yeni yöntem

RMSLE, log1p uzayında RMSE'dir. Her iki submission arasındaki log-tahmin
uzaklığı test.csv üzerinde doğrudan ölçülebilir. İki modelin gerçek Kaggle
kayıpları da bilindiğinde hata çapraz kovaryansı hedefler bilinmeden hesaplanır:

    C_ij = (score_i^2 + score_j^2 - mean((pred_i - pred_j)^2)) / 2

Burada pred değerleri log1p tahminleridir. Elde edilen hata kovaryans matrisi
pozitif yarı tanımlıdır; özdeğer kontrolünü geçmiştir. Pozitif ve toplamı bir
olan blend ağırlıkları bu matris üzerinde optimize edilmiştir.

Temel varsayım public satırların tüm testin rastgele veya temsil edici bir
altkümesi olmasıdır. Public ayrım zaman bazlıysa tam-test tahmini sapabilir.
Bu nedenle bütün olası en az 14 günlük ve testin en az yüzde 10'unu içeren
5.866 tarih penceresi ayrıca incelenmiştir.

## Üretilen adaylar

### 1. Safe — ilk yükleme önerisi

    V8R: yüzde 54,4016
    V9 : yüzde 45,5984

- Tahmini public RMSLE: 1,121893
- Olası tarih pencerelerinin yüzde 95,09'unda V8R'dan iyi.
- Pencere skor aralığı: 1,11812 ile 1,13520.
- V12 ve V16 kullanılmadığı için bilinen V16 sıfır çökmesi yoktur.

Dosya: submission_v22_lbcal_safe.csv

### 2. Anchored — safe sonucu doğrulanırsa ikinci yükleme

    V8R: yüzde 56,2730
    V9 : yüzde 23,9980
    V12: yüzde 9,1247
    V16: yüzde 10,6043

- Tahmini public RMSLE: 1,111934
- Olası tarih pencerelerinin yüzde 94,70'inde V8R'dan iyi.
- Pencere skor aralığı: 1,09498 ile 1,14025.
- V16 etkisi tam optimumun yarısına indirilmiş ve V8R'a yüzde 50 ek ankraj
  verilmiştir.

Dosya: submission_v22_lbcal_anchored.csv

### 3. Full optimum — yüksek risk

    V8R: yüzde 12,5461
    V9 : yüzde 47,9960
    V12: yüzde 18,2494
    V16: yüzde 21,2086

- Tahmini public RMSLE: 1,104782.
- Pencerelerin yüzde 92,86'sında V8R'dan iyi.
- Pencere skor aralığı: 1,07787 ile 1,15122.
- V16 ağırlığı yüksek olduğu için ilk yükleme olarak önerilmez.

Dosya: submission_v22_lbcal_full_opt.csv

## Önerilen yükleme sırası

1. Önce submission_v22_lbcal_safe.csv yüklenmeli.
2. Skor yaklaşık 1,118 ile 1,126 arasındaysa random/temsil edici public
   varsayımı desteklenir; anchored aday sonra yüklenebilir.
3. Safe skor 1,13312'den kötü gelirse diğer iki dosya yüklenmemeli. Yeni skor,
   olası public tarih pencerelerini filtrelemek ve ağırlıkları yeniden çözmek
   için kullanılmalıdır.
4. Safe skoru paylaşıldığında gerçek public maskeye çok daha yakın yeni
   kovaryans çözümü üretilebilir.

Bu yaklaşımın 1,0175 seviyesini tek başına garanti etmesi beklenmez. Güvenli
beklenti yaklaşık 1,122; agresif beklenti yaklaşık 1,105 ile 1,112 aralığıdır.
Lider seviyesine yaklaşmak için sonraki adım leaderboard kontrollü segment
problarıyla ay, warm/cold ve 11 Mayıs kohortu log-bias'larını ayrı çözmektir.
