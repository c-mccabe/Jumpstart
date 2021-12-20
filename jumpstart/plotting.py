from matplotlib import pyplot as plt
import numpy as np

plot_columns = ['In work',
                 'Years since graduation (and being on programme)',
                 'Academics',
                 'Extra-Curricular',
                 'Work Experience',
                 'Logical thinking (presentation)',
                 'Comms (presentations)',
                 'Hard work (presentation)',
                 'Problem solving (gym/spotify)',
                 'Motivations (general)',
                 'Feedback',
                 'Interview score']

plt.style.use('seaborn')


def plot_coefficients(coefficients):
    for i, col in enumerate(plot_columns):
        print(i, col)
        if col == 'Interview score':
            plt.scatter([i] * 5, [coeffienct[i] / 10 for coeffienct in coefficients], label=f'{i}. {col}')
        else:
            plt.scatter([i] * 5, [coeffienct[i] for coeffienct in coefficients], label=f'{i}. {col}')
    plt.hlines([0], *plt.xlim(), ls='--', color='black', alpha=0.4)
    #plt.legend()
    plt.title('Variable analysis', fontsize=14)
    plt.ylabel('Effect on start up interview score')
    plt.gcf().set_size_inches((16, 7))


def plot_mae(mae):
    plt.scatter(np.zeros_like(mae), mae, label='Average error')
    plt.hlines([0], *plt.xlim(), ls='--', color='black', alpha=0.4)
    plt.legend()
    plt.title('Average prediction error', fontsize=14)
    plt.gcf().set_size_inches((16, 7))

