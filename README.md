# k-fold-cross-validation
Example of regression with fixed bases of features.

While the hold out method is an intuitive approach to determining proper fiiting models, it suffers from an obvious flaw: 
Having been chosen at random, the points assigned to the training set may not adequately describe the original data. However,
we can easily extend and robustify the hold out method: k-fold cross-validation. Performing k-fold cross-validation is often the
most computationally expensive component in solving a general regresion problem. We provide a pseudo-code for applying k-fold
cross-validation.

### Algorithm：k-fold cross-validation pseudo-code
