# Portfolio

Statistical and causal inference analysis of the Diabetes 130-US Hospitals dataset, obtained from the UCI Machine Learning Repository.

**Research question:** Does HbA1c testing during hospitalisation reduce the risk of readmission in diabetic patients?

## Dataset

The raw dataset contains 101,766 inpatient encounters from 130 US hospitals (1999–2008). After preprocessing, 69,569 unique patient encounters were retained for analysis.

## Files

| File | Description |
|---|---|
| `eda_preprocessing.py` | Data cleaning, feature engineering, and exploratory visualisation |
| `item6_interaction.py` | Logistic regression with interaction term (A1C × number of diagnoses) |
| `item7.py` | GLM vs Naive Bayes comparison |
| `item8.py` | Likelihood-based model selection (AIC/BIC across nested models) |
| `item9.py` | Causal DAG and covariate adjustment via back-door criterion |
| `diabetic_data.csv` | Raw dataset |
| `diabetic_clean.csv` | Cleaned dataset (69,569 patients) |



## Reference

Clore, J.M., Cios, K., DeShazo, J.P., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
