# 👔 Employee Attrition Prediction

> Using HR analytics and machine learning to identify employees at risk of leaving — helping organizations act before it's too late.

---

## 🧩 Problem Statement

Employee attrition is one of the most expensive challenges for organizations. Replacing a single employee can cost **50–200% of their annual salary** in recruitment, training, and lost productivity. 

This project uses machine learning on an HR analytics dataset to predict which employees are most likely to leave, and identifies the key factors driving that decision — giving HR teams actionable data to improve retention.

---

## 📊 Dataset

- **Source:** IBM HR Analytics Employee Attrition dataset
- **Records:** 1,470 employees
- **Features:** Age, Department, Job Role, Job Satisfaction, Work-Life Balance, Monthly Income, Years at Company, Overtime, and more
- **Target:** `Attrition` — Yes / No

---

## 🔍 Key Findings from EDA

- **Employees doing overtime** are significantly more likely to leave — burnout is a primary driver
- **Low job satisfaction combined with low income** is the strongest predictor of attrition
- **Sales Representatives** have the highest attrition rate among all job roles
- **Employees with less than 2 years at the company** churn at a much higher rate — the first year is critical
- **Work-life balance score of 1 (Bad)** shows nearly double the attrition rate vs score 3/4

---

## 💡 Business Recommendations

1. **Monitor overtime hours closely** — employees consistently working overtime are your highest flight risk; consider workload redistribution
2. **Invest in early-tenure engagement** — create structured 6-month and 12-month check-ins for new hires
3. **Review Sales Rep compensation and career paths** — this role has disproportionately high attrition
4. **Conduct quarterly satisfaction surveys** — low satisfaction scores should trigger immediate manager intervention
5. **Improve work-life balance policies** — flexible working arrangements can directly reduce attrition in high-risk groups

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Model | Scikit-learn (Classification) |
| Evaluation | Confusion Matrix, Classification Report, ROC-AUC |
| Environment | Jupyter Notebook |

---

## 🏗️ Project Structure

```
Employee-Attrition-Prediction/
│
├── employee_attrition.ipynb   # Full EDA + model training notebook
├── dataset.csv                # HR Analytics dataset
└── README.md
```

---

## 🔬 Approach

1. **Data Cleaning** — handled missing values, removed irrelevant columns (EmployeeNumber, StandardHours)
2. **EDA** — explored relationships between attrition and all key features using visual analysis
3. **Feature Engineering** — encoded categorical variables, normalized numerical features
4. **Modelling** — trained and compared multiple classifiers, selected best performer
5. **Evaluation** — assessed model using accuracy, precision, recall, and ROC-AUC to handle class imbalance

---

## 📈 Model Insights

Key features most predictive of attrition (by feature importance):
- Overtime (Yes/No)
- Monthly Income
- Age
- Job Satisfaction
- Years at Company
- Work-Life Balance

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/jay51211/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch notebook
jupyter notebook employee_attrition.ipynb
```

---

## 👤 Author

**Jay Kumbhar**
📧 jaykumbhar518@gmail.com
💼 [LinkedIn](https://linkedin.com/in/jaykumbhar5121) | 💻 [GitHub](https://github.com/jay51211)
