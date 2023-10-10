import numpy as np

def confusion_matrix(tn, tp, fn, fp):
    conf_matrix = np.array([[tn, fp],[fn, tp]])
    return conf_matrix

def calc_confusion_matrix(y_pred, y_true):
    tn, tp, fn, fp = 0, 0, 0, 0

    for i in range(len(y_pred)):
        if y_true[i] == 0:
            if y_true[i] == y_pred[i]:
                tn += 1
            else:
                fp += 1
        
        else:
            if y_true[i] == y_pred[i]:
                tp += 1
            else:
                fn += 1
    
    conf_matrix = confusion_matrix(tn, tp, fn, fp)
    return conf_matrix