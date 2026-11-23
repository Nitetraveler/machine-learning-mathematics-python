"""Student implementation attempt for Task 1."""
import numpy as np
import pandas as pd


def impute_median(df, columns):
    # TODO: copy the DataFrame, calculate medians,
    # replace missing values and return both results.
    raise NotImplementedError


class Standardizer:
    def fit(self, X):
        # TODO: calculate column means and population std.
        # Replace zero standard deviations with 1.0.
        raise NotImplementedError

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
