"""6. Perform a regression analysis where you use a data transformation, 
a generalized linear model, or an interaction term. 
Interpret the results of your regression analysis."""

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import catppuccin
import matplotlib as mpl

mpl.style.use(catppuccin.PALETTE.latte.identifier)

df = pd.read_csv("diabetic_clean.csv")

#Logistic GLM with interaction term: A1C_tested * number_diagnoses_capped
#Hypothesis: does the protective effect of A1C testing vary by clinical complexity?
formula = ("outcome ~ A1C_tested * number_diagnoses_capped "
           "+ age_num + time_in_hospital_capped + num_medications_capped "
           "+ number_inpatient_capped + gender_bin")

model = smf.logit(formula=formula, data=df).fit()
print(model.summary2())

print("\nOdds Ratios")
odds_ratios = pd.DataFrame({
    "OR": np.exp(model.params),
    "CI_lower": np.exp(model.conf_int()[0]),
    "CI_upper": np.exp(model.conf_int()[1]),
    "p_value": model.pvalues
})
print(odds_ratios.round(4))

#Model fit
print(f"\nModel Fit")
print(f"Log-Likelihood:   {model.llf:.2f}")
print(f"AIC:              {model.aic:.2f}")
print(f"BIC:              {model.bic:.2f}")
print(f"Pseudo R-squared: {model.prsquared:.4f}")
print(f"Num observations: {int(model.nobs)}")

#Interaction plot
diag_range = np.linspace(
    df["number_diagnoses_capped"].min(),
    df["number_diagnoses_capped"].max(),
    100
)

median_vals = {
    "age_num":                 df["age_num"].median(),
    "time_in_hospital_capped": df["time_in_hospital_capped"].median(),
    "num_medications_capped":  df["num_medications_capped"].median(),
    "number_inpatient_capped": df["number_inpatient_capped"].median(),
    "gender_bin":              df["gender_bin"].median(),
}

pred_tested = pd.DataFrame({
    "A1C_tested": 1,
    "number_diagnoses_capped": diag_range,
    **median_vals
})
pred_untested = pd.DataFrame({
    "A1C_tested": 0,
    "number_diagnoses_capped": diag_range,
    **median_vals
})

prob_tested   = model.predict(pred_tested)
prob_untested = model.predict(pred_untested)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(diag_range, prob_tested   * 100, label="A1C Tested")
ax.plot(diag_range, prob_untested * 100, label="Not Tested", linestyle="--")
ax.set_xlabel("Number of Diagnoses")
ax.set_ylabel("Predicted Readmission Probability (%)")
ax.set_title("Interaction: Effect of A1C Testing by Clinical Complexity")
ax.legend()
plt.tight_layout()
plt.savefig("item6_interaction.png", dpi=150)
plt.show()