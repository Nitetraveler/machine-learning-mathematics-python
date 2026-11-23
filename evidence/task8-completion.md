# Task 8 — Production Churn Pipeline

Author: Tariq Hussain
Institution: Alnafi

## Completed implementation

- Leakage-safe scikit-learn Pipeline.
- Stratified training and test split.
- Five-fold cross-validation across six C values.
- Held-out ROC AUC evaluation.
- Scratch logistic regression comparison.
- Joblib model persistence and metadata.
- Customer scoring command-line interface.
- Default and recall-driven retention thresholds.
- Missing and nonnumeric input rejection.
- Fourteen mathematical unit tests.
- Deliberate gradient mutation and detection.
- Original implementation restoration.

## Verified results

- Best C: 0.01
- Cross-validation AUC: 0.8295170792664761
- Held-out AUC: 0.845934593459346
- Scratch AUC: 0.845934593459346
- Maximum probability difference: 8.107496538845016e-07
- Original tests: 14 passed
- Mutated tests: 1 failed, 13 passed
- Restored tests: 14 passed

## Recovery and integrity

The initial test failure was caused by an incorrect
argument list in the assisted test implementation.

The test was corrected to match the existing function
signature without changing mathematical tolerances.

Initial execution failures and recovery evidence
have been preserved.

The original gradient was restored and verified
using matching SHA256 checksums.

The implementation and tests were AI-assisted.

All Task 8 acceptance criteria passed.

Evidence: task8-validation.log
