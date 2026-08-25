# Grid-Up Datathon — Elektrik Tüketimi Tahmin Sistemi

Bu repo, elektrik dağıtım şebekesindeki 7.036 tesisin 4 aylık (Nisan-Temmuz) günlük tüketimini tahmin etmek üzere geliştirilmiş uçtan uca makine öğrenmesi modellerini, özellik mühendisliği boru hatlarını ve doğrulama protokollerini içerir.

**Hedef Metrik:** RMSLE (Root Mean Squared Log Error)  
**Test Dağılımı:** %77.84 Warm (556.319 satır) | %22.16 Cold (158.369 satır)

---

## 📁 Repo İçeriği ve Mimari

### 1. Model ve Eğitim Kodları (`train_*.py`, `build_*.py`)
* `train_v8_benchmark.py`: Public Leaderboard altın referans modelimiz (`1.13312` LB skoru).
* `train_v14_seasonal_archetypes.py`: K-Means mevsimsel tüketim arketipleri ve lokasyon hiyerarşisi.
* `train_v15_master_pipeline.py`: Fourier zaman serisi harmonikleri ve iki aşamalı CatBoost + LightGBM mimarisi.
* `train_v16_surgical_cold_start.py`: Cold-Start tesisler için Hiyerarşik Ampirik Bayes + Beta Shrinkage modeli.
* `run_v19_protocol.py`: Deterministik Güvenlik Ağı Protokolü (Safety Floor + Anomali İzolasyonu).
* `train_v20_stacker.py`: Ayrıştırılmış özellik setleriyle (Feature Disentanglement) 3'lü GBDT Stacker (LGBM + CatBoost + XGBoost).
* `train_v21_hurdle_sota.py`: İki Aşamalı Hurdle Modeli ($P_{aktif}$ Classifier + Şartlı Log-Regresör) ve Prototip Dalga Aktarımı.
* `build_v22_lb_calibrated_ensemble.py`: Leaderboard kovaryans kalibrasyonlu hibrit ensemble.

### 2. Araştırma ve Otopsi Raporları (`QWEN_*.md`, `*.md`)
* `QWEN_SOTA_ML_ARASTIRMA_RAPORU.md`: 2024-2026 akademik literatürü, Foundation modeller (Chronos, TimesFM) ve Hurdle model analizi.
* `QWEN_V18_OTOPSI_VE_KURTARMA_PLANI.md`: V16 canlı çöküşünün (0.00 hatası) kök neden analizi ve çelik zırh planı.
* `QWEN_STACKER_DEGERLENDIRMESI.md`: AutoGluon vs Özel Stacker karşılaştırması ve korelasyon kuralları.
* `V21_COHORT_COLDSTART_REPORT.md`: Kohort bazlı cold-start yönlendirme raporu.
* `V22_LEADERBOARD_CALIBRATED_REPORT.md`: Kalibre edilmiş ensemble sonuçları.

---

## 🚀 Çalıştırma

Gereksinimler:
```bash
pip install lightgbm catboost xgboost scikit-learn optuna pandas numpy
```

Örnek Model Eğitimi:
```bash
python run_v19_protocol.py
```
