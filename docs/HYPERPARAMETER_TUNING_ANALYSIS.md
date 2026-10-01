\# Hyperparameter Tuning Analysis



\## 1. Baseline Model



\### Model

DecisionTreeClassifier



\### Cross-Validation

5-fold cross-validation



\### CV F1 Macro

0.9328



\### Test Accuracy

0.9333



The baseline Decision Tree model is used as a reference point for comparing the tuned models.



\---



\## 2. Grid Search



\### Model

RandomForestClassifier



\### Search Method

Grid Search using GridSearchCV



\### Hyperparameters



\- n\_estimators = \[50, 100, 200]

\- max\_depth = \[3, 5, 10, None]

\- min\_samples\_split = \[2, 5, 10]

\- max\_features = \["sqrt", "log2"]



\### Total Combinations



72



Calculation:



3 × 4 × 3 × 2 = 72 combinations



\### Cross-Validation



5-fold



\### Total Fits



360



Calculation:



72 × 5 = 360 fits



\### Best CV F1 Macro



0.9583



\### Test Accuracy



0.9667



\---



\## 3. Random Search



\### Model

RandomForestClassifier



\### Search Method

Random Search using RandomizedSearchCV



\### Number of Iterations



30



\### Cross-Validation



5-fold



\### Total Fits



150



Calculation:



30 × 5 = 150 fits



\### Best CV F1 Macro



0.9581



\### Test Accuracy



0.9667



\---



\## 4. Comparison



| Model / Method | CV F1 Macro | Test Accuracy | Total Fits |

|---|---:|---:|---:|

| Baseline Decision Tree | 0.9328 | 0.9333 | 5 |

| Grid Search Random Forest | 0.9583 | 0.9667 | 360 |

| Random Search Random Forest | 0.9581 | 0.9667 | 150 |



Grid Search evaluates all combinations in the specified hyperparameter grid.



Random Search evaluates a selected number of combinations from the search space.



In this experiment, Grid Search required 360 model fits, while Random Search required only 150 model fits.



Both tuning approaches produced higher CV F1 Macro than the baseline model.



\---



\## 5. MLflow Tracking



All three experiments are tracked using MLflow under the experiment:



`iris-hyperparameter-tuning`



The tracked runs are:



1\. `baseline\_decision\_tree`

2\. `grid\_search\_random\_forest`

3\. `random\_search\_random\_forest`



Grid Search also produces:



`grid\_search\_all\_candidates.csv`



Random Search produces:



`random\_search\_all\_candidates.csv`



These files contain the candidate configurations and their cross-validation results.



\---



\## 6. Conclusion



The baseline Decision Tree achieved a CV F1 Macro of 0.9328.



Grid Search achieved a CV F1 Macro of 0.9583, while Random Search achieved 0.9581.



Grid Search required 360 model fits, whereas Random Search required only 150 fits. Therefore, Random Search achieved a very similar cross-validation performance with a smaller search budget.

