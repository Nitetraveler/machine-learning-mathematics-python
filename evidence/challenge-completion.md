# Challenge Extension - Newton/IRLS

Author: Tariq Hussain
Institution: Alnafi

## Implementation

Newton logistic regression using the analytic Hessian
and numpy.linalg.solve without explicit matrix inversion.

The intercept is excluded from L2 regularization.
The diagonal weight matrix is represented by a vector.

## Verified results

- Newton iterations: 7
- GD iterations: 20000
- Maximum difference versus GD: 8.881784197001e-16
- Maximum difference versus sklearn: 1.735775509437e-09
- Final gradient norm: 1.161540838981e-17
- Numerical quadratic convergence evidence: PASS
- Unit tests: 15 passed

## Scalability

A dense Hessian for 100,000 features requires about
80 GB of float64 storage before temporary allocations.

Hessian formation costs O(n*d^2).
Dense linear-system solution costs O(d^3).

## Integrity

The challenge was implemented with AI assistance.
The higher-precision GD comparison was performed
separately without modifying Task 4.

All challenge acceptance criteria passed.

Evidence: challenge-validation.log
