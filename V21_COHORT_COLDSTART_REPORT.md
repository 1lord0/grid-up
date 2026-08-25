# V21 Cohort-Aware Cold-Start Raporu

## Sonuç

V21 yalnız train'de hiç görülmeyen test trafolarını değiştirir. Leaderboard'da
1.13312 alan V8R dosyasının 556.319 warm model değeri korunur; 158.369 cold
satır geçmişte sonradan devreye alınmış trafolardan öğrenilen bir residual ile
düzeltilir. CSV yeniden yazımından sonraki warm mutlak farkı en fazla
2,33e-10'dur ve metrik açısından sıfırdır.

Üretilen ana aday:

- submission_v21_cohort_router.csv
- SHA256: 71668f3a10a5b93e4f59ad3d8f9947e5a452c42a8ccd0c47b3392561a0359c7e

## Erişim ve veri doğrulaması

İstenen TAM_SUREC_VE_RMSLE_RAPORU.md dosyası erişilebilir ve okunmuştur.
Ham veri de C:\Users\EREN\Desktop\GRİD-UP altında doğrulanmıştır.

- Train: 1.226.237 satır, 5.344 trafo, 2025-01-01 ile 2026-03-31.
- Test: 714.688 satır, 7.036 trafo, 2026-04-01 ile 2026-07-31.
- Train'de görülmeyen test trafoları: 2.024 trafo ve 158.369 satır.
- Bu trafoların 1.326'sı 2026-05-11 tarihinde birlikte görünmeye başlar.
- Train tam sıfırları: 57.536 satır, yüzde 4,692.
- Train'de ilk ve son kayıt aralığına göre tahmini 69.816 eksik trafo-gün
  vardır; 1.240 trafoda en az bir boşluk görülür.
- Test cold panelinde bir günden büyük yalnız 153 ardışık-tarih boşluğu vardır.
  Eksik-gün sinyali bu nedenle V21'e alınmadı; testte desteği çok düşüktür.

Yarışma sahibinin açıklaması doğrultusunda sıfırlar gerçek tüketimsizlik veya
devre dışılık durumudur. Train-unseen trafolar yeni devreye alınmıştır. V21 bu
iki bilgiyi doğrudan model tasarımına taşır.

## Önceki modellerde bulunan riskler

1. Fold A cold tanımı sadece cutoff öncesinde görülmeme koşuluna dayanıyordu.
   Devreye alınmadan beri geçen süre ve toplu devreye alma kohortu yoktu.
2. Satır bazlı rastgele KFold, aynı trafonun ya da aynı commissioning olayının
   günlük satırlarını train ve validation'a dağıtabilir. Bu gerçek cold-start
   davranışını ölçmez.
3. V17 target encoding satır bazlı KFold ile üretildi. Trafo kimliği doğrudan
   özellik olmasa bile aynı operasyonel olayın komşu satırları iki tarafta
   kalabildiğinden iyimserlik riski vardır.
4. Cold tahminlere pozitif safety floor uygulamak, gerçek sıfır açıklamasıyla
   çelişir. Fold A cold satırlarının yalnız yüzde 4'ü sıfır olmasına rağmen
   2.5*guc tabanının karesel log-hatasının yaklaşık yüzde 53,8'i bu satırlardan
   gelir.
5. Tek Fold A üzerinde threshold ve blend ağırlığı aramak seçim yanlılığı
   yaratır. V21 tüm commissioning tarihini dışarıda bırakan GroupKFold ve dört
   rolling-origin fold kullanır.
6. Public leaderboard cold kaliteyi tek başına kanıtlamaz. Public/private
   ayrımı zaman bazlıysa 11 Mayıs toplu kohortu public skorunda az temsil
   edilebilir veya hiç görünmeyebilir.

## Model

Eğitim evreni veri başlangıcındaki sol-kesilmiş trafolar değildir. İlk haftadan
sonra veri setine giren 3.193 trafo ve 362.580 satır kullanılır.

Hedef:

    log1p(tuketim) - log1p(2.5 * guc)

Cold-safe özellikler:

- güç, log-güç ve güç sınıfı;
- il, ilçe ve bölge;
- ay, haftanın günü ve yıllık Fourier bileşenleri;
- hedef panelden gözlenebilen ilk görünme tarihi;
- devreye alınmadan beri geçen gün ve yaş haftası;
- aynı gün devreye alınan toplam, ilçe, bölge ve güç kırılımı trafo sayıları.

Trafo geçmiş hedefi, lag, trafo target encoding'i ve gelecekteki hedef bilgisi
kullanılmaz.

Ham residual doğrudan uygulanmaz. Düzeltme ağırlığı aşağıdaki güven kapısıyla
belirlenir:

    weight = 0.50 * exp(-(abs(log_delta) / 0.90) ^ 2)

Bu kapı, rolling validation'da kararsız olduğu görülen büyük sapmaları söndürür.
Submission'da residual V8R cold tahmininin logaritmasına eklenir; V8R'ın mevcut
coğrafi ve kapasite kalibrasyonu korunur.

## Davranışsal validasyon

Referans taban aşağıdaki bütün ölçümlerde 2.5*guc'tur.

| Validasyon | Taban | Güven kapılı V21 | Kazanç | Ham V21 |
|---|---:|---:|---:|---:|
| Fold A, commissioning-date GroupKFold | 1,811266 | 1,794709 | +0,016557 | 1,769145 |
| Rolling 2025-03-31 -> 2025-07-31 | 1,811266 | 1,808644 | +0,002622 | 2,195249 |
| Rolling 2025-06-30 -> 2025-09-30 | 1,986561 | 1,966821 | +0,019740 | 1,910605 |
| Rolling 2025-09-30 -> 2025-12-31 | 2,080233 | 2,061706 | +0,018527 | 2,085112 |
| Rolling 2025-12-31 -> 2026-03-31 | 1,660077 | 1,652204 | +0,007873 | 2,521478 |
| Tüm tarihsel commissioning-date CV | 2,016941 | 1,997877 | +0,019064 | 1,935622 |

Güven kapılı sürüm bütün rolling fold'larda tabanı iyileştirir. Ham modelin son
rolling fold'daki 2,521478 çöküşü, testlerin geçmesinin veya tek bir CV skorunun
yeterli olmadığını gösterir ve kapının neden gerekli olduğunu davranışsal olarak
kanıtlar.

Sıfır ve pozitif hedefleri ayıran hurdle model ayrıca denendi. Aynı trafoların
birlikte dışarıda bırakıldığı ölçümde doğrudan model 1,7243, hurdle model 2,0073
verdi. Bu nedenle mimari olarak çekici görünmesine rağmen V21'e alınmadı.

## Submission bütünlüğü

- Toplam satır: 714.688.
- Değişen cold satır: 158.369.
- Model değeri korunan warm satır: 556.319.
- Cold trafo: 2.024.
- V8R cold medyanı: 1.430,07.
- V21 cold medyanı: 1.226,22.
- Ortalama gate ağırlığı: 0,3367.
- NaN, sonsuz veya negatif tahmin: 0.
- Cold tahmin içinde tam sıfır: 0. Bu bir floor nedeniyle değil, log-residual
  model ve V8R tabanının sürekli tahmin üretmesi nedeniyledir.

Satır bazlı inceleme dosyası v21_cold_diagnostics.csv olarak üretilmiştir.

## Çalıştırma

Derin davranışsal validasyon:

    python train_v21_cohort_router.py validate
      --data-dir "C:\Users\EREN\Desktop\GRİD-UP"
      --deep
      --report v21_validation_results.json

Submission üretimi:

    python train_v21_cohort_router.py submit
      --data-dir "C:\Users\EREN\Desktop\GRİD-UP"
      --base-submission submission_v8r_verified_final.csv
      --output submission_v21_cohort_router.csv
      --diagnostics v21_cold_diagnostics.csv
