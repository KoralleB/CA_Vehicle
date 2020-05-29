import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def roc_plot(label,fpr,tpr,roc_auc):
    plt.figure()
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], label = label[i]+' AUC = %0.2f' % roc_auc[i])
    plt.plot([0, 1],[0, 1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def pr_plot(label,recall,precision):
    plt.figure()
    for i in range(len(label)):
        plt.plot(recall[i], precision[i], label = label[i])
    plt.xlim([0,1])
    plt.ylim([0,0.2])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc='upper right')
    plt.show()

def conf_mat(cm):
    cm = pd.DataFrame(cm)
    cm.columns = ['ICEV', 'EV']
    cm.rename(index={0: 'ICEV', 1: 'EV'}, inplace=True)
    return(cm)

def var_imp(ind_i):
    with open('../export_files/features.p', 'rb') as fp:
        features = pickle.load(fp)
    importances = np.load('../export_files/importances_rf.npy')

    # df of importance
    d = {'features': features, 'importances': importances}
    imp_df = pd.DataFrame(d)
    imp_df = imp_df.sort_values('importances', ascending=False)
    imp_df = imp_df.reset_index(drop=True)

    plt.title('Feature Importances')
    plt.barh(range(ind_i), imp_df['importances'][:ind_i], color='b', align='center')
    plt.yticks(range(ind_i), [imp_df['features'][i] for i in range(ind_i)])
    plt.xlabel('Relative Importance')
    plt.show()
