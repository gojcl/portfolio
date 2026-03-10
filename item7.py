"""7. Perform statistical inference on one or more model parameters by fitting 
two different statistical models to the same dataset. 
Of those, one should be a (possibly generalized) linear model,
whereas the other should be a mathematical model, 
probabilistic model or a simulation model. 
Compare the conclusions obtained by both models and evaluate their agreement."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catppuccin
import matplotlib as mpl
import statsmodels.formula.api as smf
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

mpl.style.use(catppuccin.PALETTE.latte.identifier)

df = pd.read_csv("diabetic_clean.csv")

features = [
    "A1C_tested", "age_num", "time_in_hospital_capped",
    "num_medications_capped", "number_diagnoses_capped",
    "number_inpatient_capped", "gender_bin"
]

#Model 1 -> logistic GLM 
formula = ("outcome ~ A1C_tested + age_num + time_in_hospital_capped "
           "+ num_medications_capped + number_diagnoses_capped "
           "+ number_inpatient_capped + gender_bin")

glm = smf.logit(formula=formula, data=df).fit()
print("Logistic GLM")
print(glm.summary2())

glm_or = pd.DataFrame({
    "OR": np.exp(glm.params),
    "CI_lower": np.exp(glm.conf_int()[0]),
    "CI_upper": np.exp(glm.conf_int()[1]),
    "p_value": glm.pvalues
}).drop("Intercept")
print("\nGLM Odds Ratios:")
print(glm_or.round(4))

#Model 2 -> naive bayes
X = df[features]
y = df["outcome"]

nb = GaussianNB()
nb.fit(X, y)

nb_probs = nb.predict_proba(X)[:, 1]

print("\nNaive Bayes")
print(f"Class priors: Not readmitted={nb.class_prior_[0]:.3f}, Readmitted={nb.class_prior_[1]:.3f}")
print(f"\nFeature means per class (theta):")
theta_df = pd.DataFrame(nb.theta_, columns=features,
                         index=["Not Readmitted", "Readmitted"])
print(theta_df.round(3))

#Comparison
glm_probs = glm.predict(df)

comparison = pd.DataFrame({
    "GLM_prob": glm_probs,
    "NB_prob": nb_probs,
    "outcome": y
})

print("\nModel Comparison")
print(comparison.describe().round(4))
print(f"\nCorrelation between predicted probabilities: "
      f"{comparison['GLM_prob'].corr(comparison['NB_prob']):.4f}")

#Visualisation - ROC curves
fpr_glm, tpr_glm, _ = roc_curve(y, glm_probs)
fpr_nb, tpr_nb, _ = roc_curve(y, nb_probs)
auc_glm = auc(fpr_glm, tpr_glm)
auc_nb = auc(fpr_nb, tpr_nb)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(fpr_glm, tpr_glm, label=f"GLM (AUC = {auc_glm:.3f})")
ax.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_nb:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves: GLM vs Naive Bayes")
ax.legend()

plt.tight_layout()
plt.savefig("item7_roc.png", dpi=150)
plt.show()