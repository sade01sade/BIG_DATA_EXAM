# BIG_DATA_EXAM
# 💳 Credit Card Fraud Detection — Big Data Analytics Capstone Project

This project investigates credit card fraud using customer transaction behavior data. The goal is to explore patterns in fraudulent vs. legitimate transactions, build machine learning models, and present interactive visual analytics using Power BI.

---

## 📁 Dataset

- **Source**: [ULB Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) *(accessed externally, not via Kaggle)*
- **Rows**: 284,807 transactions
- **Features**: 30 anonymized input features (V1–V28), `Time`, `Amount`, and `Class` (fraud = 1, legit = 0)

---

## 🛠️ Tools & Technologies

| Tool           | Purpose                              |
|----------------|--------------------------------------|
| Google Colab   | Python-based data cleaning, SMOTE    |
| pandas, seaborn, matplotlib | Data manipulation and visualization |
| Power BI       | Interactive data dashboards          |
| GitHub         | Project repository                   |

---

## 📊 Power BI Dashboard

The dashboard includes:

- **KPI Cards**:
  - 🧮 Total Transactions
  - 🔴 Total Fraud Transactions
  - ⚠️ Fraud Rate (%)
- **Bar Chart**:
  - Class distribution by Train/Test set
- **Box/Violin Plot** *(imported image)*:
  - `norm_amount` distribution by fraud status
- **Scatter Plot**:
  - Feature interactions (e.g., `V1` vs `V2`) colored by class
- **Histograms**:
  - Distribution of `V1`, `V14`, etc.
- **Slicers**:
  - Filter by Class (Fraud / Legit)
  - Filter by Set (Train / Test)
  - Filter by transaction amount range

---

## 🧼 Data Preprocessing (Python)

1. Loaded and explored data using pandas
2. Normalized `Amount` and `Time` using `StandardScaler`
3. Applied **SMOTE** to balance fraud and legit cases in training data
4. Split data into training and testing sets (80/20)

Cleaned datasets (`train_resampled.csv`, `test_set.csv`, `combined_data.csv`) were exported and used in Power BI.

---

## 🧠 Insights & Observations

- Fraud transactions have distinct feature patterns in `V1`, `V14`, and `Amount`
- Legit transactions dominate the dataset (~99.8%)
- Using SMOTE significantly improves model training
- Visual filters make it easier to isolate and study fraud patterns

---

## 📝 How to Use

1. Clone or download this repo
2. Open `combined_data.csv` in Power BI
3. Use the slicers and visuals to explore fraud insights
4. (Optional) Run preprocessing code in the Jupyter/Colab notebook

---

## 📸 Sample Dashboard Screenshot

![Dashboard](images/dashboard_screenshot.png)  
*(Replace with your actual screenshot path)*

---

## 👨‍💻 Author

- **Name**: SADE GEORGE SADE  
- **Institution**: AUCA (Adventist University of Central Africa)  
- **Course**: INSY8314 – Web Design / Big Data Analytics  
- **Supervisor**: Mr. BYIRINGIRO Eric

---

## 📂 Repository Structure

