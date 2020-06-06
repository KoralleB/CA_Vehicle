import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display


def roc_plot(label, fpr, tpr, roc_auc):
    """
    plot ROC curve for selected models
    :param label: legend label. input: ['model1', 'model2',...]
    :param fpr: fpr of model. input: list of p.load('../output_files/fpr_modelname.npy')
    :param tpr: tpr of model. input: list of p.load('../output_files/tpr_modelname.npy')
    :param roc_auc: roc_auc of model. input: list of p.load('../output_files/roc_auc_modelname.npy')
    """
    plt.figure()
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], label=label[i] + ' AUC = %0.2f' % roc_auc[i], alpha=0.75)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def pr_plot(label, recall, precision):
    """
    plot PR curve for selected models
    :param label: legend label. input: ['model1', 'model2',...]
    :param recall: recall of model. input: list of p.load('../output_files/recall_modelname.npy')
    :param precision: precision of model. input: list of p.load('../output_files/precision_modelname.npy')
    """
    plt.figure()
    for i in range(len(label)):
        plt.plot(recall[i], precision[i], label=label[i], alpha=0.75)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc='upper right')
    plt.show()


def conf_mat(label, cm):
    """
    confusion matrix pandas dataframe for selected models
    :param label: pandas title. input: ['model1', 'model2',...]
    :param cm: confusion matrix of model. input: list of p.load('../output_files/cm_modelname.npy')
    """
    for i in range(len(label)):
        cm_pd = pd.DataFrame(data=cm[i],
                             index=['ICEV', 'EV'],
                             columns=['ICEV', 'EV'])
        cm_pd = cm_pd.style.set_caption(label[i])
        display(cm_pd)


def acc_print(label, acc):
    """
    accuracy value for selected models
    :param label: model accuracy. input: ['model1', 'model2',...]
    :param acc: accuracy of model. input: list of p.load('../output_files/acc_modelname.npy')
    """
    for i in range(len(label)):
        print("Accuracy of ", label[i], " is ", np.round(acc[i], 5))


def roc_print(label, y_true, y_pred):
    """
    fpr and tpr (ROC) values for selected models
    :param label: model accuracy. input: ['model1', 'model2',...]
    :param y_true: prediction of model. input: p.load('../output_files/y_true.npy')
    :param y_pred: prediction of model. input: list of p.load('../output_files/y_pred_modelname.npy')
    """
    for i in range(len(label)):
        FP = np.logical_and(y_true != y_pred[i], y_pred[i] == 1).sum()
        FN = np.logical_and(y_true != y_pred[i], y_pred[i] == 0).sum()
        TP = np.logical_and(y_true == y_pred[i], y_true == 1).sum()
        TN = np.logical_and(y_true == y_pred[i], y_true == 0).sum()
        FPR = 1. * FP / (FP + TN)
        TPR = 1. * TP / (TP + FN)
        PPV = 1. * TP / (TP + FP)
        print("FPR of ", label[i], " is ", np.round(FPR, 5))
        print("TPR of ", label[i], " is ", np.round(TPR, 5))
        print("PPV of ", label[i], " is ", np.round(PPV, 5))
        print('')

def var_imp(modelname, ind_i):
    """
    variable importance plot for a selected random forest model
    :param modelname: model name. input: 'log', 'log_over', "rf_smote', 'svm_under', etc
    :param ind_i: number of top important variables
    """
    with open('../output_files/features.p', 'rb') as fp:
        features = pickle.load(fp)

    path_load = '../output_files/importances_' + modelname + '.npy'
    importances = np.load(path_load)

    # df of importances
    d = {'features': features, 'importances': importances}
    imp_df = pd.DataFrame(d)
    imp_df = imp_df.sort_values('importances', ascending=False)
    imp_df = imp_df.reset_index(drop=True)

    plt.title('Feature Importances')
    plt.barh(range(ind_i), imp_df['importances'][:ind_i], color='b', align='center')
    plt.yticks(range(ind_i), [imp_df['features'][i] for i in range(ind_i)])
    plt.xlabel('Relative Importance')
    plt.show()


def train_nn(train_nn_results, label, title, yaxis):
    """
    plot loss or accuracy vs epoch of nn models.
    :param train_nn_results: epoch training output. input: list of
    np.load('../output_files/train_loss_results_nn_smote_x.npy') or train_accuracy_results_nn_smote_x.npy
    :param label: legend label. input: ['model1', 'model2',...]
    :param title: plot title. input: 'title'
    :param yaxis: y axis label. input: 'yaxis'
    """
    plt.figure(figsize=(12,5))
    for i in range(len(label)):
        plt.plot(train_nn_results[i], label=label[i], alpha=0.75)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(yaxis)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
