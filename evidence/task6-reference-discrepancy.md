# Task 6 — Reference Experiment Discrepancy

Author: Tariq Hussain
Institution: Alnafi

## Environment

- NumPy 1.26.4
- SciPy 1.13.1
- scikit-learn 1.5.1

## Official experiment

The supplied reference implementation was executed
with its prescribed random seed, sample counts,
noise parameters, polynomial degrees and lambda grid.

## Independently verified results

- Selected polynomial degree: 5
- Selected lambda: 0.1
- Scratch test MSE: 0.1561991018883563
- sklearn test MSE: 0.1561991018883562
- Maximum prediction difference: 2.6645352591003757e-15
- Required test MSE: below 0.15

## Finding

The scratch implementation agrees with scikit-learn
to floating-point precision.

The official reference experiment exceeds its own
stated test MSE acceptance threshold.

## Evidence

- task6-independent-diagnostic.log
- task6-original/
- task6-refined-diagnostic.log

## Integrity

The original reference results and implementation
have been preserved.

A supplementary finer lambda grid was evaluated
using training-only five-fold cross-validation.

No grading thresholds were modified.
No official JSON results were manually altered.

The exercise correction was AI-assisted, and the
initial unfinished exercise has been preserved.

## Supplementary experiment results

- Lambda candidates: 141
- Selected lambda: 0.1
- Best cross-validation MSE: 0.10112285595338606
- Refined test MSE: 0.1561991018883563
- Official threshold satisfied: No

The refined training-only search independently selected
the same lambda as the original eight-candidate search.

The official threshold remains unchanged and unmet.

## Final disposition

Implementation verified against scikit-learn.
Original experiment and results preserved.
Official MSE acceptance criterion remains unresolved.
Instructor clarification is required.

Technical validation evidence:
task6-final-technical-validation.log
