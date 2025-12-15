                                                   Smart Spam Detection

Beyond Accuracy: This study shifts spam detection evaluation from just accuracy to statistical rigor, generalization, and confidence reliability.

Hybrid Ensemble Advantage: Combining diverse learners + modern boosters delivers robust detection, balancing performance and trustworthiness.

Stacking Shines in Reliability:Stacking with Logistic Regression achieves 95% accuracy and excellent calibration (ECE = 0.0119), making it ideal for risk-sensitive applications.

XGBoost = Accuracy Champion:Best single model with 95.8% accuracy, proving that modern gradient boosting remains a powerful standalone solution.

Voting Paradox Discovered:Hard Voting and Soft Voting yield identical accuracy, yet Soft Voting improves recall and ROC-AUC, revealing hidden benefits of probabilistic decisions.

Calibration Matters:Low ECE values (< 0.02) confirm that ensemble outputs are confidence-aware, not just correct—critical for automated email filtering.

Statistical Validation Ensured:10-fold cross-validation + significance testing confirms model stability and avoids overfitting bias.

Interpretability + Trust:Ensemble methods enhance decision transparency, supporting deployment in confidence-critical security systems.

Key Insight:When thresholds are fixed, probability-based and discrete decisions may align—but calibrated probabilities still matter.
