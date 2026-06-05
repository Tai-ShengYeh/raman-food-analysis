# 拉曼光譜食品分析 — Orange Data Mining 工作流
# Raman Spectroscopy Food Analysis — Orange Data Mining Workflow

一個**可直接開啟、零程式碼**的 Orange Data Mining 工作流，用真實開源拉曼光譜資料做食用油的**真偽鑑別**與**氧化品質預測**。附可重現的 Python 對照腳本與真實結果。

A ready-to-open, **zero-code** Orange Data Mining workflow that uses real open Raman spectra to perform edible-oil **authentication** and **oxidation-quality prediction**, with a reproducible Python counterpart and verified results.

---

## 📦 內容 / Files
| 檔案 | 說明 |
|------|------|
| `raman_food_workflow.ows` | **Orange 工作流**（File → Spectra → Preprocess Spectra → PCA / 分類 / PLS）。用 Orange 3 + Orange-Spectroscopy 開啟 |
| `run_raman_analysis.py` | **可重現 Python 對照**（scikit-learn）：ALS 去基線 → 裁切 → SNV → PCA → SVM/RF 分類 → PLS 迴歸，產生下方所有圖與 `results.json` |
| `data/Raman2.csv` | 真實拉曼光譜（215 樣本 × 1044 波段，15 種油，含 Class + PeroxideValue）。來源 Mendeley `ctgg7k4m5g`（CC BY 4.0） |
| `data/OilClassKey.csv` | 油種代碼對照表（1=初榨橄欖油 … 19=核桃油） |
| `figures/` | 4 張真實結果圖 |
| `results.json` | 鎖定的真實數值 |

---

## ▶️ 怎麼用 Orange 開啟 / How to run in Orange

```bash
pip install orange3 orange-spectroscopy   # 需要 Python >= 3.10
python -m Orange.canvas               # 開啟 Orange，再 File > Open 載入 .ows
```
開啟 `raman_food_workflow.ows` 後：
1. 點 **File** widget → 載入 `data/Raman2.csv`。把 **Class** 設為 target（分類）或把 **PeroxideValue** 設為 target（迴歸）；另一欄設為 meta。
2. 雙擊每個 widget 看結果。畫布上的文字框已標註每一步要設定什麼。

---

## 🔬 工作流的科學 / The pipeline (and why)

```
File (Raman2.csv)
 ├─ Spectra ………………… 看 RAW：拉曼光譜有傾斜的「螢光背景」駝峰
 └─ Preprocess Spectra（拉曼專用順序）
      1. Spike Removal ……………… 去宇宙射線尖峰
      2. Baseline Correction → Rubber band (Positive) … 去螢光背景 ★拉曼最關鍵
      3. Cut → 400–1800 cm⁻¹ ……… 只留指紋區
      4. Normalize Spectra → SNV … 消除強度/散射差異
        ├─ Spectra ……………… 看預處理後：背景被拉平、峰變清楚
        ├─ PCA → Scatter Plot … 真偽/油種分群
        ├─ Test & Score ← SVM / Random Forest → Confusion Matrix … 分類（target=Class）
        └─ PLS（Model 類別）→ Test & Score ………… 迴歸 R²/RMSE（target=PeroxideValue）
```

> **為什麼拉曼跟 NIR 不一樣？** 拉曼光譜常被**螢光背景**淹沒，所以「去基線（Baseline Correction / Rubber band 或 Asymmetric Least Squares）」是拉曼**最關鍵**的預處理步驟——這正是 Orange Spectroscopy 的招牌功能。NIR 通常不需要這步。

> ⚠️ **PLS widget 用哪個？** 請用 **Model 類別**的 PLS widget（`Orange.widgets.model.owpls`）。Spectroscopy 外掛裡那顆 PLS 已被標示 deprecated（會在未來版本移除）。Model 類別的 PLS 是「學習器（Learner）」，所以接法跟 SVM/RF 一樣：PLS 的 **Learner 輸出 → Test & Score 的 Learner 輸入**，再由 Test & Score 給出交叉驗證的 R²/RMSE（記得把 target 設成 PeroxideValue）。

---

## 📊 真實結果 / Verified results
（由 `run_raman_analysis.py` 在 `data/Raman2.csv` 上實際執行；可重現）

| 分析 | 結果 |
|------|------|
| 資料 | 215 樣本、15 種油、1044 波段（18–2088 cm⁻¹）、指紋區 708 波段 |
| PCA | PC1 = **87.4%**、PC1+PC2 = **93.8%**；橄欖油與其他油自然分群 |
| **真偽鑑別**（橄欖油 vs 其他油，SVM） | 測試集 **98.5%**、5-fold CV **86.5%**（RF 90.8%） |
| **氧化迴歸**（PLS 預測過氧化價） | 全部油種 R² = **0.585**；**單一 EVOO 內 R² = 0.936**（RMSE 2.46 meq/kg） |

### 🎓 教學重點 / Key lesson
分類（真偽）容易；**定量迴歸的陷阱在「混淆變因」**：跨 15 種油預測過氧化價只有 R²≈0.59，因為光譜變異被「油種」主導；但**先控制油種、只看初榨橄欖油**，R² 跳到 **0.94**——拉曼確實能測氧化/新鮮度，但你得先控制混淆因子。

---

## 🖼️ Figures
| 檔案 | 內容 |
|------|------|
| `figures/raman_fig1_baseline.png` | RAW 光譜的螢光背景 + ALS 基線；預處理後乾淨指紋 |
| `figures/raman_fig2_pca.png` | PCA 解釋變異 + 橄欖油 vs 其他油散布 |
| `figures/raman_fig3_authentication.png` | 橄欖油真偽鑑別混淆矩陣 |
| `figures/raman_fig4_peroxide.png` | PLS 過氧化價迴歸（全部油 vs 單一 EVOO） |

## 🔁 重現 / Reproduce
```bash
pip install spectral scikit-learn scipy numpy matplotlib pandas
python run_raman_analysis.py   # 讀 data/Raman2.csv，輸出 figures/ 與 results.json
```

## 📚 資料來源 / Data source & license
Edible-oil Raman/IR dataset — Mendeley Data **`ctgg7k4m5g`** (v2), CC BY 4.0.
Zhao, Zhan, Xu et al. (University of Nebraska–Lincoln). https://data.mendeley.com/datasets/ctgg7k4m5g/2
工具 / tools: Orange 3 + Orange-Spectroscopy · scikit-learn · scipy。
