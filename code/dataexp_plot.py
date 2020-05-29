import matplotlib.pyplot as plt
import numpy as np


def cat_plot(data_column, title, yticklabels):
    """
    column: df.colname
    title: 'title'
    yticklabels: [list,of,labels]
    """
    fig, ax = plt.subplots(1, len(data_column), figsize=(9, 3))
    plt.subplots_adjust(wspace=0.8)
    width = 0.75

    for i in range(len(data_column)):
        a = data_column[i].value_counts().sort_index()
        x = list(a.index)
        y = list(a)

        ind = np.arange(len(y))
        ax[i].barh(ind, y, width)
        ax[i].set_yticks(ind)
        ax[i].set_yticklabels(yticklabels[i])
        ax[i].set_title(title[i])

    plt.show()