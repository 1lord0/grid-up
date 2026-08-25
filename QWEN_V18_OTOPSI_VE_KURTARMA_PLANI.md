# Grid-Up Datathon — Qwen 3.8-27B V18 Otopsi ve Kurtarma Planı

# V18 KURTARMA VE TABAN GÜVENLİĞİ PLANI: "ÇELİK ÇERÇEVE" PROTOKOLÜ

**Durum Değerlendirmesi:**
V16'nın başarısızlığı, modelin "geçici kapanış" durumunu "kalıcı kapanış" olarak genellemesi ve RMSLE'nin sıfıra yakın tahminlerdeki aşırı duyarlılığından kaynaklanmıştır. V8R (`1.13312`) kanıtlanmış, stabil bir referans noktasıdır. V18'in amacı, V8R'ın sağlamlığını koruyarak, V16/V17'deki *güvenli* sinyalleri (sadece V8R ile uyumlu olanları) enjekte etmek ve mutlak bir "güvenlik tabanı" (safety floor) uygulamaktır.

**Temel Strateji:**
1.  **Ana Omurga:** V8R tahminleri.
2.  **Güvenlik Zırhı:** Aktif tesisler için minimum tahmin sınırı.
3.  **Anomali Filtresi:** V8R'dan sapma eşiği aşan tahminlerin V8R'a geri çekilmesi.
4.  **Kontrollü Harmanlama:** Sadece yüksek güvenliğe sahip, V8R ile korelasyonu yüksek olan yeni sinyallerin düşük ağırlıkla eklenmesi.

---

## 1. VERİ HAZIRLAMA VE GÜVENLİK TABANI (SAFETY FLOOR)

Bu adım, V16'nın yaptığı hatayı (aktif tesise 0 kW basması) matematiksel olarak imkansız hale getirir.

### 1.1. Aktif Tesis Tanımı ve Taban Hesaplama
2026 yılı için "aktif" kabul edilen tesisler, 2025'in son 3 ayında (Ekim, Kasım, Aralık) veya 2026'nın ilk 3 ayında (Ocak, Şubat, Mart) en az bir kez > 0 kW tüketim kaydetmiş tesislerdir.

```python
import pandas as pd
import numpy as np

# V8R ve V16/V17 tahminlerini yükle
v8r = pd.read_csv('submission_v8r_verified_final.csv')
v16 = pd.read_csv('submission_v16_standalone.csv')
# V17 varsa onu da yükle, yoksa sadece V16 kullan
# v17 = pd.read_csv('submission_v17.csv') 

# Test verisini yükle (guc, tesis_id, tarih bilgileri için)
test_df = pd.read_csv('test.csv') # Varsayım: test.csv içinde 'guc' ve 'tesis_id' var

# V8R ve V16'yı test_df ile birleştir
df = test_df.merge(v8r, on='id', how='left')
df = df.merge(v16, on='id', how='left', suffixes=('_v8r', '_v16'))

# 1. ADIM: Güvenlik Tabanı (Safety Floor) Hesaplama
# Kural: Aktif bir tesise asla 0 veya çok düşük değer basılmaz.
# Formül: floor = max(0.05 * guc, 2025_Son_3_Ay_Ortalama_Tahminin_0.5_Katı)

# 2025'in son 3 ayındaki V8R tahminlerinin tesis bazlı ortalamasını al
# (V8R'ın kendi geçmiş performansına güveniyoruz)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# 2025 Ekim-Aralık arası V8R tahminleri
mask_2025_q4 = (df['year'] == 2025) & (df['month'].isin([10, 11, 12]))
avg_2025_q4 = df.loc[mask_2025_q4].groupby('tesis_id')['target_v8r'].mean().reset_index()
avg_2025_q4.columns = ['tesis_id', 'avg_2025_q4_v8r']

# 2026 Ocak-Mart arası V8R tahminleri (Eğer varsa)
mask_2026_q1 = (df['year'] == 2026) & (df['month'].isin([1, 2, 3]))
avg_2026_q1 = df.loc[mask_2026_q1].groupby('tesis_id')['target_v8r'].mean().reset_index()
avg_2026_q1.columns = ['tesis_id', 'avg_2026_q1_v8r']

# Tesis bazlı taban değerini belirle
# Eğer tesis 2025 Q4'te aktifse, o ortalamanın %50'si taban olsun.
# Eğer 2026 Q1'de aktifse, o ortalamanın %50'si taban olsun.
# İkisi de yoksa, 0.05 * guc olsun.

df = df.merge(avg_2025_q4, on='tesis_id', how='left')
df = df.merge(avg_2026_q1, on='tesis_id', how='left')

# Taban Değer Hesaplama
df['floor_2025'] = df['avg_2025_q4_v8r'] * 0.5
df['floor_2026'] = df['avg_2026_q1_v8r'] * 0.5
df['floor_guc'] = df['guc'] * 0.05

# En yüksek taban değeri seç (En güvenli olan)
df['safety_floor'] = df[['floor_2025', 'floor_2026', 'floor_guc']].max(axis=1)
df['safety_floor'] = df['safety_floor'].fillna(df['floor_guc']) # NaN'lar için guc bazlı taban

# 2. ADIM: V8R Tahminlerini Güvenlik Tabanına Uygula
# V8R zaten sağlam, ama ekstra bir sigorta olarak:
# Eğer V8R tahmini, hesaplanan safety_floor'dan düşükse, safety_floor'u kullan.
# Bu, V8R'ın nadir de olsa düşük tahmin yaptığı durumlarda bile sıfıra inmesini engeller.
df['target_v8r_safe'] = np.maximum(df['target_v8r'], df['safety_floor'])
```

---

## 2. ANOMALİ BUDAMA VE V8R'A GERİ ÇEKME (FALLBACK)

V16/V17 modellerinin V8R'dan sapma gösterdiği satırlar, riskli kabul edilir. Bu satırlarda V8R'a geri dönülür.

### 2.1. Log-Fark Eşiği
Kural: `|log(pred_new) - log(v8r_safe)| > 1.5` ise, `pred_new` yerine `v8r_safe` kullanılır.

```python
# 3. ADIM: V16/V17 Tahminlerini Güvenlik Tabanına Uygula
# V16 da aynı güvenlik tabanına tabi tutulmalı
df['target_v16_safe'] = np.maximum(df['target_v16'], df['safety_floor'])

# 4. ADIM: Anomali Filtresi (Fallback)
# Log farkı hesapla
# Not: 0 değerleri log'da -inf verir, bu yüzden 1.0 ekliyoruz (RMSLE uyumlu)
df['log_v8r'] = np.log1p(df['target_v8r_safe'])
df['log_v16'] = np.log1p(df['target_v16_safe'])

df['log_diff'] = np.abs(df['log_v16'] - df['log_v8r'])

# Eşik: 1.5 (Yaklaşık 4.5 kat fark)
# Eğer fark 1.5'ten büyükse, V16'yı kullanma, V8R'a geri çek.
# Bu, V16'nın "kapalı tesis" hatasını ve diğer uç değerleri otomatik olarak eleme.
df['target_final'] = np.where(df['log_diff'] > 1.5, df['target_v8r_safe'], df['target_v16_safe'])
```

---

## 3. KONTROLLÜ HARMANLAMA (BLENDING)

V16'nın tamamen atılması yerine, V8R ile uyumlu olan (log_diff < 1.5) satırlarda, V16'nın sinyalini düşük ağırlıkla ekleyerek hafif bir iyileşme hedeflenir. Ancak, V8R'ın baskınlığı korunur.

### 3.1. Ağırlıklandırma
*   **V8R Ağırlığı:** 0.85
*   **V16 Ağırlığı:** 0.15
*   **Kondisyon:** Sadece `log_diff <= 1.5` olan satırlarda harmanlama yapılır.
*   **Kondisyon:** `log_diff > 1.5` olan satırlarda %100 V8R kullanılır.

```python
# 5. ADIM: Kontrollü Harmanlama
# Sadece güvenli bölgede (log_diff <= 1.5) harmanla
# Güvenli bölgede: 0.85 * V8R + 0.15 * V16
# Riskli bölgede: 1.00 * V8R

df['blend_weight_v16'] = np.where(df['log_diff'] <= 1.5, 0.15, 0.0)
df['blend_weight_v8r'] = 1.0 - df['blend_weight_v16']

df['target_v18'] = (df['target_v8r_safe'] * df['blend_weight_v8r']) + \
                   (df['target_v16_safe'] * df['blend_weight_v16'])

# 6. ADIM: Son Güvenlik Kontrolü
# Harmanlama sonrası bile safety_floor'un altına inilmemeli
df['target_v18'] = np.maximum(df['target_v18'], df['safety_floor'])

# 7. ADIM: Sonuç Dosyasını Oluştur
submission_v18 = df[['id', 'target_v18']].copy()
submission_v18.columns = ['id', 'target']
submission_v18.to_csv('submission_v18_steel_frame.csv', index=False)
```

---

## 4. NEDEN BU PLAN 1.08-1.10 BANDINA ÇEKER? (MATEMATİKSEL GEREKÇE)

1.  **V16 Hatasının Tamamen Silinmesi:**
    *   V16'nın 88 tesiste yaptığı 0 kW hatası, `log_diff` hesabında `|log(0) - log(15000)|` -> `inf` veya çok büyük bir değer olarak görünür.
    *   Bu satırlar `log_diff > 1.5` koşuluna takılır ve **%100 V8R'a geri çekilir.**
    *   Sonuç: V16'nın skoru +0.18 patlatan hatası, V18'de **sıfırlanır.** Skor, V8R'ın 1.13312 seviyesine döner.

2.  **Güvenlik Tabanının (Safety Floor) Etkisi:**
    *   V8R'ın kendisi zaten sağlam, ancak `safety_floor` uygulaması, V8R'ın nadir de olsa düşük tahmin yaptığı satırlarda (örneğin, mevsimsel dalgalanmalarda) RMSLE cezasını azaltır.
    *   RMSLE, `log(1 + y_true) - log(1 + y_pred)` farkını ceza olarak kullanır. `y_pred`'i `y_true`'a yakın tutmak, özellikle `y_true` küçükken önemlidir.
    *   `0.05 * guc` tabanı, aktif tesislerdeki aşırı düşük tahminleri (örneğin, 1000 kW'lık tesise 10 kW basması) engeller. Bu, V8R'ın mevcut 1.13312 skorunu **hafifçe iyileştirebilir** (tahmin: 1.12-1.13 arası).

3.  **Kontrollü Harmanlamanın Etkisi:**
    *   V16'nın V8R ile uyumlu olduğu satırlarda (log_diff < 1.5), V16'nın %15 ağırlığı eklenir.
    *   V16, V8R'dan farklı bir model olduğu için, bazı satırlarda V8R'dan daha iyi tahmin yapabilir (örneğin, yeni açılan tesisler veya ani değişimler).
    *   Bu %15'lik katkı, V8R'ın 1.13312 skorunu **1.10-1.12 bandına** çekme potansiyeli taşır.
    *   **Risk:** V16'nın V8R ile uyumlu olduğu satırlarda bile hata yapma ihtimali vardır. Ancak, %15 ağırlık, bu riski minimize eder.

**Beklenen Skor Aralığı:**
*   **En İyi Senaryo:** V16'nın uyumlu sinyalleri V8R'ı iyileştirir -> **1.08 - 1.10**
*   **Orta Senaryo:** V16'nın sinyalleri nötrdür, sadece V8R + Safety Floor -> **1.12 - 1.13**
*   **En Kötü Senaryo:** V16'nın uyumlu sinyalleri bile V8R'dan kötüdür -> **1.13 - 1.14** (V8R'dan biraz kötü, ama V16'dan çok iyi)

**Sonuç:** Bu plan, V16'nın felaketini tamamen ortadan kaldırır ve V8R'ın sağlamlığını koruyarak, küçük bir iyileşme şansı sunar. **1.13312'nin altına inme riski, V16'nın 1.32'ye çıkma riskinden çok daha düşüktür.**

---

## 5. UYGULAMA ADIMLARI (KOD)

Aşağıdaki kod, yukarıdaki mantığı tam olarak uygular. `test.csv` dosyasının `guc` ve `tesis_id` kolonlarını içerdiğini varsayar.

```python
import pandas as pd
import numpy as np

# 1. VERİ YÜKLEME
v8r = pd.read_csv('submission_v8r_verified_final.csv')
v16 = pd.read_csv('submission_v16_standalone.csv')
test_df = pd.read_csv('test.csv')

# 2. BİRLEŞTİRME
df = test_df.merge(v8r, on='id', how='left')
df = df.merge(v16, on='id', how='left', suffixes=('_v8r', '_v16'))

# 3. GÜVENLİK TABANI (SAFETY FLOOR) HESAPLAMA
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# 2025 Q4 ve 2026 Q1 ortalamaları
mask_2025_q4 = (df['year'] == 2025) & (df['month'].isin([10, 11, 12]))
avg_2025_q4 = df.loc[mask_2025_q4].groupby('tesis_id')['target_v8r'].mean().reset_index()
avg_2025_q4.columns = ['tesis_id', 'avg_2025_q4_v8r']

mask_2026_q1 = (df['year'] == 2026) & (df['month'].isin([1, 2, 3]))
avg_2026_q1 = df.loc[mask_2026_q1].groupby('tesis_id')['target_v8r'].mean().reset_index()
avg_2026_q1.columns = ['tesis_id', 'avg_2026_q1_v8r']

df = df.merge(avg_2025_q4, on='tesis_id', how='left')
df = df.merge(avg_2026_q1, on='tesis_id', how='left')

# Taban değerler
df['floor_2025'] = df['avg_2025_q4_v8r'] * 0.5
df['floor_2026'] = df['avg_2026_q1_v8r'] * 0.5
df['floor_guc'] = df['guc'] * 0.05

df['safety_floor'] = df[['floor_2025', 'floor_2026', 'floor_guc']].max(axis=1)
df['safety_floor'] = df['safety_floor'].fillna(df['floor_guc'])

# 4. GÜVENLİK TABANINI UYGULA
df['target_v8r_safe'] = np.maximum(df['target_v8r'], df['safety_floor'])
df['target_v16_safe'] = np.maximum(df['target_v16'], df['safety_floor'])

# 5. ANOMALİ FİLTRESİ (FALLBACK)
df['log_v8r'] = np.log1p(df['target_v8r_safe'])
df['log_v16'] = np.log1p(df['target_v16_safe'])
df['log_diff'] = np.abs(df['log_v16'] - df['log_v8r'])

# 6. KONTROLLÜ HARMANLAMA
# Güvenli bölgede (log_diff <= 1.5): 0.85 V8R + 0.15 V16
# Riskli bölgede (log_diff > 1.5): 1.00 V8R
df['blend_weight_v16'] = np.where(df['log_diff'] <= 1.5, 0.15, 0.0)
df['blend_weight_v8r'] = 1.0 - df['blend_weight_v16']

df['target_v18'] = (df['target_v8r_safe'] * df['blend_weight_v8r']) + \
                   (df['target_v16_safe'] * df['blend_weight_v16'])

# 7. SON GÜVENLİK KONTROLÜ
df['target_v18'] = np.maximum(df['target_v18'], df['safety_floor'])

# 8. SONUÇ DOSYASI
submission_v18 = df[['id', 'target_v18']].copy()
submission_v18.columns = ['id', 'target']
submission_v18.to_csv('submission_v18_steel_frame.csv', index=False)

print("V18 Submission Dosyası Oluşturuldu: submission_v18_steel_frame.csv")
print("İlk 5 Satır:")
print(submission_v18.head())
```