# python -m analysis_accuracy_helpers --mini 0 --maxi 1000 &
# python -m analysis_accuracy_helpers --mini 1000 --maxi 2000 &
# python -m analysis_accuracy_helpers --mini 2000 --maxi 3000 &
# python -m analysis_accuracy_helpers --mini 3000 --maxi 4000 &
# python -m analysis_accuracy_helpers --mini 4000 --maxi 5000 &
# python -m analysis_accuracy_helpers --mini 5000 --maxi 6000 &
# python -m analysis_accuracy_helpers --mini 6000 --maxi 7000 &
# python -m analysis_accuracy_helpers --mini 7000 --maxi 8000 &
# python -m analysis_accuracy_helpers --mini 8000 --maxi 9000 &
# python -m analysis_accuracy_helpers --mini 9000 --maxi 10000 &
# python -m analysis_accuracy_helpers --mini 10000 --maxi 11000 &
# python -m analysis_accuracy_helpers --mini 11000 --maxi 12000 &
# python -m analysis_accuracy_helpers --mini 12000 --maxi 13000 &
# python -m analysis_accuracy_helpers --mini 13000 --maxi 14000 &
# python -m analysis_accuracy_helpers --mini 14000 --maxi 15000 &
# python -m analysis_accuracy_helpers --mini 15000 --maxi 16000 &
# python -m analysis_accuracy_helpers --mini 16000 --maxi 17000 &
# python -m analysis_accuracy_helpers --mini 17000 --maxi 18000 &
# python -m analysis_accuracy_helpers --mini 18000 --maxi 19000 &
# python -m analysis_accuracy_helpers --mini 19000 --maxi 20000 &
# python -m analysis_accuracy_helpers --mini 20000 --maxi 21000 &
# python -m analysis_accuracy_helpers --mini 21000 --maxi 22000 &
# python -m analysis_accuracy_helpers --mini 22000 --maxi 23000 &
# python -m analysis_accuracy_helpers --mini 23000 --maxi 24000 &
# python -m analysis_accuracy_helpers --mini 24000 --maxi 25000 &
# python -m analysis_accuracy_helpers --mini 25000 --maxi 26000 &
# python -m analysis_accuracy_helpers --mini 26000 --maxi 27000 &
# python -m analysis_accuracy_helpers --mini 27000 --maxi 28000 &

import argparse
import os
import timeit
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
from analysis_clustering_helpers import get_cvfold_extended
from analysis_tree_helpers_2 import HTree

from scipy.cluster.hierarchy import cut_tree,linkage
from scipy.optimize import linear_sum_assignment 
from sklearn.metrics import adjusted_rand_score
from pprint import pprint
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mini",default=0,type=int)
parser.add_argument("--maxi",default=100,type=int)


def main(mini=0,maxi=100):
    base_path = str(Path.home())+'/Dropbox/AllenInstitute/CellTypes/dat/'
    result_path = base_path +   'result/hctest/'
    cvsets_pth    = base_path + 'result/patchseq_v4_leafonly/'
    metadata_file = base_path + 'raw/patchseq-v4/PS_v4_beta_0-4.mat'
    htree_file    = base_path + 'raw/patchseq-v4/dend_RData_Tree_20181220.csv'

    csval='0-1'
    matfile   = sio.loadmat(metadata_file,squeeze_me=True)
    file_list = sorted([file for file in os.listdir(cvsets_pth) if 'cs_'+csval+'_' in file])
    file_list = [file for file in file_list if '.mat' in file]
    _,_,train_leaf,val_leaf = get_cvfold_extended(cvfile=cvsets_pth+file_list[0],refdata=matfile)
    Z = np.concatenate([train_leaf['E_z'],val_leaf['E_z']],axis=0)
    col = np.concatenate([train_leaf['color'],val_leaf['color']],axis=0)
    labels_true = np.concatenate([train_leaf['labels'],val_leaf['labels']],axis=0)

    #Full inhibitory tree
    htree = HTree(htree_file=htree_file)
    inh_subtree = htree.get_subtree(node='n59')
    inh_descendant_dict = inh_subtree.get_all_descendants(leafonly=False)

    import pickle
    PIK = base_path+'/result/hierarchy_search_cs_'+csval+'/best_subclass_lists_cpl_str_'+csval+'.dat'
    #Read previously saved best_node_lists_dict_acc and best_node_lists_dict_ari:
    with open(PIK, "rb") as f:
        X=pickle.load(f)

    #Sort all classifications by number of classes. This is to avoid duplicate hierarchical tree cuts
    class_list = []
    n_list = ['n76','n61','n90','n109']
    for elem0 in X[0][n_list[0]]:
        for elem1 in X[0][n_list[1]]:
            for elem2 in X[0][n_list[2]]:
                for elem3 in X[0][n_list[3]]:
                    class_list.append(elem0[0]+elem1[0]+elem2[0]+elem3[0])
    class_list.sort(key=len)
    
    #Adding Chodl explicitly to the classifications - the list is modified in-place. 
    [c.extend(['Sst Chodl']) for c in class_list]

    n_classes = []
    acc_list = []

    #Get complete hierarchical tree for the representations (Ward clustering):
    ind = np.isin(labels_true,inh_descendant_dict['n59'])
    labels_true_class = labels_true[ind]
    Z_class = Z[ind,:]
    dend = linkage(Z_class, 'ward')


    n_classes = 0
    n_classes_list = []
    acc_list = []
    i_list = []

    for i,classes in enumerate(class_list):
        if mini<=i and i<maxi:
            #Update ground truth labels:
            current_labels_true = labels_true.copy()
            for node in classes:
                current_labels_true[np.isin(current_labels_true,inh_descendant_dict[node])] = node

            #Obtain hierarchical clustering labels:
            if n_classes != len(classes):
                n_classes = len(classes)
                labels_pred = cut_tree(dend, n_clusters=n_classes)
                labels_pred = np.ravel(labels_pred)    
            else:
                pass

            hc_labels = np.unique(labels_pred)
            gt_labels = np.unique(current_labels_true)
            Cij = np.zeros((hc_labels.size,gt_labels.size))
            for i,hc_label in enumerate(hc_labels):
                for j,gt_label in enumerate(gt_labels):
                    Cij[i,j] = np.sum(np.logical_and(labels_pred==hc_label,
                                                    current_labels_true==gt_label))

            Cij = -1*Cij #Because linear_sum_assignment minimizes cost.
            row_ind, col_ind = linear_sum_assignment(Cij)
            Cij = -1*Cij #Convert back
            acc = Cij[row_ind, col_ind].sum()/current_labels_true.size

            acc_list.append(acc)
            n_classes_list.append(n_classes)
            i_list.append(i)

    PIK = base_path+'result/hierarchy_search_cs_'+csval+'/'+'min_'+str(mini)+'max_'+str(maxi)+'.dat'
    data = {'acc_list':acc_list,
            'n_classes_list':n_classes_list,
            'i_list':i_list}
    with open(PIK, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))