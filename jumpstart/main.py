from jumpstart.preprocessing import get_X_y
from jumpstart.plotting import plot_coefficients, plot_mae
from jumpstart.linearregression import get_regression_coefficients

import pandas as pd


def main():
    df = pd.read_excel('jumpstart.xlsx', index_col=0)
    X, y = get_X_y(df)
    coefficients, mae = get_regression_coefficients(X, y)
    plot_coefficients(coefficients)
    #plot_mae(mae)


if __name__ == '__main__':
    main()
