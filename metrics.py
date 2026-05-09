import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(outputs, targets):

    y_reg_pred = outputs[:, 0]
    y_reg_true = targets[:, 0]
    mse = mean_squared_error(y_reg_true, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
    r2 = r2_score(y_reg_true, y_reg_pred)

    ci = get_cindex(y_reg_true, y_reg_pred)

    y_reg_cls = (y_reg_pred >= -6.0).astype(int)
    
    y_cls_prob = 1 / (1 + np.exp(-outputs[:, 1]))  
    y_cls_pred = (y_cls_prob > 0.5).astype(int)
    y_cls_true = targets[:, 1].astype(int)
    
    accuracy = accuracy_score(y_cls_true, y_cls_pred)
    precision = precision_score(y_cls_true, y_cls_pred, zero_division=0)
    recall = recall_score(y_cls_true, y_cls_pred, zero_division=0)
    f1 = f1_score(y_cls_true, y_cls_pred, zero_division=0)
    
    auc = 0
    if len(np.unique(y_cls_true)) > 1:
        auc = roc_auc_score(y_cls_true, y_cls_prob)
    
    consistency = np.mean(y_reg_cls == y_cls_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'ci': ci, 
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'consistency': consistency 
    }

def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair != 0:
        return summ/pair
    else:
        return 0

def rmse(y,f):
    return np.mean((np.array(y) - np.array(f)) ** 2)

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))