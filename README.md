
# Customer Churn Prediction System (Telecom)

## Project Overview
This project predicts whether a telecom customer is likely to churn using Machine Learning.

It is based on the research paper:
A Survey on Churn Analysis in Various Business Domains

The system helps telecom companies identify high-risk customers and reduce customer loss.

---

## Features
- Interactive Data Analysis Dashboard using Streamlit
- Machine Learning Model using Random Forest
- Single Customer Prediction
- Bulk CSV Upload Prediction
- Churn Probability Output

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Imbalanced-learn (SMOTE)
- Joblib

---

## Dataset
Telco Customer Churn Dataset (Kaggle)
~7043 records, 21 features

Includes:
- Customer demographics
- Service usage details
- Contract and billing information
- Target variable: Churn (Yes/No)

---

## How to Run the Project

git clone https://github.com/YOUR_USERNAME/churnanalysis.git
cd churnanalysis

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python train_model.py

streamlit run streamlit_app.py

---

## Output
- Churn Prediction (Yes/No)
- Churn Probability (%)
- Bulk prediction CSV download

---

## Key Insights
- Month-to-month contract customers churn the most
- High monthly charges increase churn risk
- Low tenure customers are more likely to leave

---

## Business Impact
- Reduces customer churn
- Improves retention strategies
- Supports CRM decision-making
- Increases customer lifetime value

---


## Conclusion
This project demonstrates how Machine Learning helps telecom companies reduce churn and make better business decisions.
