from matplotlib import pyplot as plt
import numpy as np

plot_columns = [
    'In work',
    'Years since graduation (and being on programme)',
    'Academics',
    'Extra-Curricular',
    'Work Experience',
    'Motivations (general)',
    'Problem solving (gym/spotify)',
    'Jumpstart interview score',
    '3rd sector', 'Clueless grad', 'Consulting/Banking', 'Other',
    'Set on startups'
]


def plot_coefficients(coefficients):
    [plt.scatter([i] * 5, [coef[i] for coef in coefficients], label=col) for i, col in enumerate(plot_columns[0:7])]
    plt.scatter([7] * 5, [coef[7] / 10 for coef in coefficients], label='Jumpstart interview score')
    plt.legend()
    plt.hlines([0], *plt.xlim(), ls='--', color='black', alpha=0.4)
    plt.title('Variable analysis', fontsize=14)
    plt.ylabel('Effect on start up interview score')
    plt.gcf().set_size_inches((16, 7))


def plot_mae(mae):
    plt.scatter(np.zeros_like(mae), mae, label='Average error')
    plt.hlines([0], *plt.xlim(), ls='--', color='black', alpha=0.4)
    plt.legend()
    plt.title('Average prediction error', fontsize=14)
    plt.gcf().set_size_inches((16, 7))

