# BIG_DATA_EXAM

## 👤 Author

* **Name**: SADE GEORGE SADE
* **ID**: 26915
* **Institution**: AUCA
* **Module**: INSY8314 — Big Data Analytics
* **Instructor**: Mr. BYIRINGIRO Eric

---

# 💳 Credit Card Fraud Detection — Big Data Analytics Capstone Project

This project aims to detect fraudulent credit card transactions using machine learning and big data visualization techniques. The pipeline includes data exploration, preprocessing (including SMOTE for class imbalance), and an interactive Power BI dashboard.

---

## 📁 Dataset

- **Source**: ULB Credit Card Fraud Detection Dataset  
- **Rows**: 284,807 transactions  
- **Features**: 
  - `Time`, `Amount` (continuous)
  - `V1` to `V28` (anonymized via PCA)
  - `Class` → Target: 0 = Legit, 1 = Fraud

---

## 🛠️ Tools & Technologies

| Tool           | Purpose                              |
|----------------|--------------------------------------|
| Python (Google Colab) | Data cleaning, balancing (SMOTE) |
| Power BI       | Interactive dashboard & analytics    |
| GitHub         | Project version control              |
| pandas, matplotlib, seaborn, scikit-learn | Python data stack |

---

## 🧼 Data Cleaning & Preprocessing (Python Code)

### ▶️ Step 1: Load and Preview the Data

```python
import pandas as pd

df = pd.read_csv("creditcard.csv")
print(df.shape)
print(df.head())
````

---

### ▶️ Step 2: Check for Missing Values

```python
print(df.isnull().sum())
```

---

### ▶️ Step 3: Normalize `Amount` and `Time`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['norm_amount'] = scaler.fit_transform(df[['Amount']])
df['norm_time'] = scaler.fit_transform(df[['Time']])

# Drop original columns
df = df.drop(['Amount', 'Time'], axis=1)
```
---
### ▶️ Step 4: Visual checks 

```python
  import matplotlib.pyplot as plt
import seaborn as sns

# Plot Amount vs Class
plt.figure(figsize=(6,4))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount by Class')
plt.show()

# Plot Time distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Time'], bins=50, kde=True)
plt.title('Transaction Time Distribution')
plt.show()
```
## Screenshots of visual checks

<img width="415" height="287" alt="Screenshot 2025-08-02 153619" src="https://github.com/user-attachments/assets/db0f1e7a-1810-453f-bb68-d8602dd0cb77" />

<img width="418" height="293" alt="Screenshot 2025-08-02 153658" src="https://github.com/user-attachments/assets/063ace02-095e-4ef6-a54e-c56622ccff4b" />



---

### ▶️ Step 5: Split into Train and Test Sets

```python
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

### ▶️ Step 6: Apply SMOTE to Balance the Training Set

```python
!pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(y_train_resampled.value_counts())
```

---

### ▶️ Step 7: Export for Power BI

```python
# Combine features and label
train_df = pd.concat([pd.DataFrame(X_train_resampled, columns=X.columns),
                      pd.Series(y_train_resampled, name='Class')], axis=1)

test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

# Add 'Set' column for dashboard filtering
train_df['Set'] = 'Train'
test_df['Set'] = 'Test'

# Combine into one CSV
combined_df = pd.concat([train_df, test_df])
combined_df.to_csv('combined_data.csv', index=False)
```

---

## 📊 Power BI Dashboard

Includes:

* ✅ KPI Cards: Total Transactions, Fraud Count, Fraud Rate
* 📊 Bar Chart: Class count by Train/Test
* 📈 Violin/Box Plot: norm\_amount by Class (exported from Python)
* 📉 Histograms: Feature distributions (e.g., V1, V14)
* 🎯 Scatter Plots: V1 vs V2, sized by norm\_amount
* 🎚️ Slicers: ClassLabel, Set, norm\_amount (range slider)

---

## 🧠 Key Insights

* Class imbalance is extreme: <0.2% fraud
* Fraudulent transactions show distinct patterns in V1, V14, V17
* SMOTE helps balance data and train effective models
* Power BI slicers allow interactive fraud exploration

---

## 📝 Repository Structure

```
📦 credit-card-fraud-project
├── data/
│   └── combined_data.csv
├── notebook/
│   └── fraud_cleaning.ipynb
├── visuals/
│   └── violin_plot.png
├── dashboard/
│   └── fraud_dashboard.pbix
└── README.md
```

---


## 🏁 Conclusion

This project demonstrates how real-world financial fraud can be analyzed using open data, Python, and business intelligence tools. The combination of SMOTE balancing, feature scaling, and visual storytelling enables users to better understand and detect fraudulent behavior.

---

### Power BI dashboard screenshot

<img width="581" height="331" alt="Screenshot 2025-08-02 154212" src="https://github.com/user-attachments/assets/f508b13e-d3f7-42a0-9c22-b962795b8f45" />


### 🔁 You can find the cleaned data, the PowerPoint and the pbiviz file in the link Below 👇:
```
  https://drive.google.com/drive/folders/1iIEg00OkaLQPOSzN51An1GrDUT9jh9Xq?usp=drive_link
```

### THANK YOU 
