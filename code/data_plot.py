import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cat_plot(data_column, title, yticklabels):
    """
    plotting two categorical variables in two separated plots.
    :param data_column: categorical column names to plot. input: [df.colname1, df.colname2]
    :param title: title of plots. input: ['title1', 'title2']
    :param yticklabels: categorical label names. input: [['label_1x', 'label_1y', ...], ['label_2x', 'label_2y', ...]]
    """
    fig, ax = plt.subplots(1, len(data_column), figsize=(9, 3))
    plt.subplots_adjust(wspace=0.8)
    width = 0.75

    for i in range(len(data_column)):
        a = data_column[i].value_counts().sort_index()
        y = list(a)

        ind = np.arange(len(y))
        ax[i].barh(ind, y, width)
        ax[i].set_yticks(ind)
        ax[i].set_yticklabels(yticklabels[i])
        ax[i].set_title(title[i])

    plt.show()


def cat_plot_y(data, colname, title, label, i, j):
    """
    plot a categorical variable with two levels which are numerical.
    :param data: df name. input: df
    :param colname: categorical column name. input: 'colname'
    :param title: plot title. input: 'title'
    :param label: categorical label names. input: ['label_1', 'label_2']
    :param i: the first level of the categorical variable. input: 1
    :param j: the second level of the categorical variable. input: 2
    """
    one_serie = data.groupby(colname)['y'].value_counts().sort_index()[i]
    two_serie = data.groupby(colname)['y'].value_counts().sort_index()[j]
    one_serie = one_serie * (100 / one_serie.sum())
    two_serie = two_serie * (100 / two_serie.sum())
    df_per = pd.DataFrame(np.array([list(one_serie), list(two_serie)]), index=[i, j], columns=[0, 1])
    values = [df_per.values[0][0], df_per.values[1][0], df_per.values[0][1], df_per.values[1][1]]

    kx = -0.1
    ky = -0.02

    table = pd.crosstab(data[colname], data['y'])
    ax = table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

    for i, rec in enumerate(ax.patches):
        ax.text(rec.get_xy()[0] + rec.get_width() / 2 + kx, rec.get_xy()[1] + rec.get_height() / 2 + ky,
                '{:.1%}'.format(values[i] / 100), fontsize=12, color='black', backgroundcolor='white')

    plt.title(title)
    plt.xticks(np.arange(2), label, rotation=360)
    plt.legend(bbox_to_anchor=(1, 1.02), labels=['ICEV', 'EV'])
    plt.xlabel('')
    plt.show()
