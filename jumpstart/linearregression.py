from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error


def fit_regression(X, y):
    model = LinearRegression()
    model.fit(np.nan_to_num(X), y)
    y_pred = model.predict(np.nan_to_num(X))
    coefficients = model.coef_
    mae = mean_absolute_error(y, y_pred)
    return coefficients, mae


def get_regression_coefficients(X, y):
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    coefficients_list = []
    mae_list = []

    for train_index, test_index in kf.split(X):
        coefficients, mae = fit_regression(X[train_index, :], y[train_index])
        coefficients_list.append(coefficients)
        mae_list.append(mae)

    return coefficients_list, mae_list
