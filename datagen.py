import pdb
import random

import numpy as np
import scipy.io as sio
from tensorflow.python.keras.utils import Sequence


def dataset_50fold(exp_name='',cvset=0):
    from pathlib import Path   
    from sklearn.model_selection import KFold
    
    dir_pth = {}
    curr_path = str(Path().absolute()) + '/'
    dir_pth['data'] = curr_path + 'data/raw/'
    dir_pth['result'] = curr_path + 'data/results/' + exp_name + '/'
    dir_pth['logs'] = dir_pth['result'] + 'logs/'
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True) 
    
    data = sio.loadmat(dir_pth['data'] + 'PS_v2_beta_0-4.mat', squeeze_me=True)
    
    X1 = data['T_dat']
    X2 = data['E_dat']
    X1_ispaired = data['T_ispaired']==1
    X2_ispaired = data['E_ispaired']==1
    X1_labels = data['cluster']

    #Ignore pairing between cells that are not mapped to leaf nodes in the transcriptomic data
    #Note: X1[:n,:] and X2[:n,:] are the cells in X1 and X2 for which both data modalities were observed
    leaf_inds = np.flatnonzero(np.logical_and(data['T_ispaired']==1, data['cluster_color']!='#808080'))
    X1_inds = np.arange(X1.shape[0])
    X2_inds = np.arange(X2.shape[0])
    X1_nonleaf_inds = np.setdiff1d(X1_inds,leaf_inds)
    X2_nonleaf_inds = np.setdiff1d(X2_inds,leaf_inds)
    X1_ispaired[X1_nonleaf_inds] = False
    X2_ispaired[X2_nonleaf_inds] = False

    #Initialize 
    X1_inds = np.arange(X1.shape[0])
    X2_inds = np.arange(X2.shape[0])
    X1_ispaired = X1_ispaired.reshape(X1_ispaired.size,1)
    X2_ispaired = X2_ispaired.reshape(X2_ispaired.size,1)

    #Split into 50 training and validation sets; respect the .
    paired_ind = np.flatnonzero(X1_ispaired)
    kf = KFold(n_splits=50, shuffle = True, random_state=10)
    folds = list(kf.split(paired_ind))

    #Separate training and validation inds
    test_set_inds = folds[cvset][1].copy()
    val_ind = paired_ind[test_set_inds]
    train_ind_T = np.setdiff1d(train_ind_T,val_ind)
    train_ind_E = np.setdiff1d(train_ind_E,val_ind)

    #Assemble training dataset:
    train_X1 = X1[train_ind_T,:]
    train_X2 = X2[train_ind_E,:]

    train_ispaired_T = ispaired_T[train_ind_T]
    train_ispaired_E = ispaired_E[train_ind_E]
    train_paired_ind_T = np.where(train_ispaired_T)[0]
    train_paired_ind_E = np.where(train_ispaired_E)[0]
    train_all_ind_T = np.arange(train_X1.shape[0])
    train_all_ind_E = np.arange(train_X2.shape[0])

    #Assemble validation dataset:
    val_X1 = X1[val_ind,:]
    val_X2 = X2[val_ind,:]

    val_ispaired_T = ispaired_T[val_ind]
    val_ispaired_E = ispaired_E[val_ind]
    val_paired_ind_T = np.where(val_ispaired_T)[0]
    val_paired_ind_E = np.where(val_ispaired_E)[0]
    val_all_ind_T = np.arange(val_X1.shape[0])
    val_all_ind_E = np.arange(val_X2.shape[0])

    #Combine into dictionary consumed by the generator
    train_dat = {'T': train_X1, 'T_ispaired': train_ispaired_T, 'T_paired_ind': train_paired_ind_T, 'T_all_ind': train_all_ind_T, 
                 'E': train_X2, 'E_ispaired': train_ispaired_E, 'E_paired_ind': train_paired_ind_E, 'E_all_ind': train_all_ind_E}

    val_dat = {'T': val_X1, 'T_ispaired': val_ispaired_T, 'T_paired_ind': val_paired_ind_T, 'T_all_ind': val_all_ind_T, 
               'E': val_X2, 'E_ispaired': val_ispaired_E, 'E_paired_ind': val_paired_ind_E, 'E_all_ind': val_all_ind_E}
    return train_dat, val_dat, train_ind_T, train_ind_E, val_ind, dir_pth


def dataset_50foldSF(exp_name='',cvset=0, fname='dat2_4k',ngenes=5000,val_perc = 0.1):
    from pathlib import Path   
    from sklearn.model_selection import KFold
    
    dir_pth = {}
    curr_path = str(Path().absolute())
    if '/Users/fruity' in curr_path:
        base_path = '/Users/fruity/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/split-facs/'
    elif '/home/rohan' in curr_path:
        base_path = '/home/rohan/Dropbox/AllenInstitute/CellTypes/'
        dir_pth['data'] = base_path + 'dat/raw/split-facs/'
    elif '/allen' in curr_path:
        base_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Rohan/'
        dir_pth['data'] = base_path + 'dat/raw/split-facs/'
    else: #beaker relative paths
        base_path = '/'

    dir_pth['result'] =     base_path + 'dat/result/' + exp_name + '/'
    dir_pth['checkpoint'] = dir_pth['result'] + 'checkpoints/'
    dir_pth['logs'] =       dir_pth['result'] + 'logs/'
    
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True) 
    Path(dir_pth['checkpoint']).mkdir(parents=True, exist_ok=True) 

    data = sio.loadmat(dir_pth['data'] + fname +'.mat', squeeze_me=True)
    if type(data['clusters_TE_unpaired'])==str:
        data['clusters_TE_unpaired'] = [data['clusters_TE_unpaired']]
    if type(data['clusters_T_only'])==str:
        data['clusters_T_only'] = [data['clusters_T_only']]
    if type(data['clusters_E_only'])==str:
        data['clusters_E_only'] = [data['clusters_E_only']]
    
    dat_T      = data['T_dat']
    dat_E      = data['E_dat']
    ispaired_T = data['T_ispaired']==1
    ispaired_E = data['E_ispaired']==1
    
    #Initialize 
    train_ind_T = np.arange(dat_T.shape[0])
    train_ind_E = np.arange(dat_E.shape[0])
    ispaired_T = ispaired_T.reshape(ispaired_T.size,1)
    ispaired_E = ispaired_E.reshape(ispaired_E.size,1)

    #Split into 50 training and validation sets.     
    paired_ind = np.where(ispaired_T)[0]
    val_ind_T = []
    val_ind_E = []
    
    np.random.seed(seed=cvset)
    for c in data['clusters_T_only'].tolist()+data['clusters_TE_unpaired']:
        inds = np.where(data['T_cluster']==c)[0]
        val_ind_T.extend(np.random.choice(inds, size=round(inds.size*val_perc), replace=False))
    for c in data['clusters_E_only'].tolist()+data['clusters_TE_unpaired']:
        inds = np.where(data['E_cluster']==c)[0]
        val_ind_E.extend(np.random.choice(inds, size=round(inds.size*val_perc), replace=False))
        
    T_unpaired_ind = np.setdiff1d(train_ind_T,paired_ind)
    val_ind_T.extend(np.random.choice(T_unpaired_ind, size=round(T_unpaired_ind.size*val_perc), replace=False))

    E_unpaired_ind = np.setdiff1d(train_ind_E,paired_ind)
    val_ind_E.extend(np.random.choice(E_unpaired_ind, size=round(E_unpaired_ind.size*val_perc), replace=False))

    val_ind_T = np.unique(np.array(val_ind_T))
    val_ind_E = np.unique(np.array(val_ind_E))
    
    #Separate training and validation inds
    train_ind_T = np.setdiff1d(train_ind_T,val_ind_T)
    train_ind_E = np.setdiff1d(train_ind_E,val_ind_E)

    #Assemble training dataset:
    train_dat_T = dat_T[train_ind_T,:ngenes]
    train_dat_E = dat_E[train_ind_E,:ngenes]
    
    train_ispaired_T = ispaired_T[train_ind_T]
    train_ispaired_E = ispaired_E[train_ind_E]
    train_paired_ind_T = np.where(train_ispaired_T)[0]
    train_paired_ind_E = np.where(train_ispaired_E)[0]
    train_all_ind_T = np.arange(train_dat_T.shape[0])
    train_all_ind_E = np.arange(train_dat_E.shape[0])

    #Assemble validation dataset:
    val_dat_T = dat_T[val_ind_T,:ngenes]
    val_dat_E = dat_E[val_ind_E,:ngenes]

    val_ispaired_T = ispaired_T[val_ind_T]
    val_ispaired_E = ispaired_E[val_ind_E]
    val_paired_ind_T = np.where(val_ispaired_T)[0]
    val_paired_ind_E = np.where(val_ispaired_E)[0]
    val_all_ind_T = np.arange(val_dat_T.shape[0])
    val_all_ind_E = np.arange(val_dat_E.shape[0])

    #Combine into dictionary consumed by the generator
    train_dat = {'T': train_dat_T, 'T_ispaired': train_ispaired_T, 'T_paired_ind': train_paired_ind_T, 'T_all_ind': train_all_ind_T, 
                 'E': train_dat_E, 'E_ispaired': train_ispaired_E, 'E_paired_ind': train_paired_ind_E, 'E_all_ind': train_all_ind_E}

    val_dat = {'T': val_dat_T, 'T_ispaired': val_ispaired_T, 'T_paired_ind': val_paired_ind_T, 'T_all_ind': val_all_ind_T, 
               'E': val_dat_E, 'E_ispaired': val_ispaired_E, 'E_paired_ind': val_paired_ind_E, 'E_all_ind': val_all_ind_E}
    return train_dat, val_dat, train_ind_T, train_ind_E, val_ind_T, val_ind_E, dir_pth


class DatagenTE(Sequence):
    '''Creates a valid batch from a pool of paired and unpaired T and E data. 
        \n Inputs:
        \n`dataset`: dict with `T` and `E` fields
        \n`n_paired_per_batch`: number of paired T and E datapoints introduced per batch
    '''

    def __init__(self, dataset, batch_size, n_paired_per_batch, steps_per_epoch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_paired_per_batch = n_paired_per_batch
        self.steps_per_epoch = steps_per_epoch
        assert steps_per_epoch>0, "steps_per_epoch is less than 1"
        return

    def __len__(self):
        'Return number of batches per epoch'
        # This is the exit condition for the generator
        return self.steps_per_epoch

    def __getitem__(self, idx):
        'Generate one batch of data'
        T_batch, E_batch, ispaired = self.pop_batch()
        return ({'in_ae_0': T_batch, 'ispaired_ae_0': ispaired,
                 'in_ae_1': E_batch, 'ispaired_ae_1': ispaired},
                {'ou_ae_0': T_batch, 'ld_ae_0': np.zeros((T_batch.shape[0], 2)),
                 'ou_ae_1': E_batch, 'ld_ae_1': np.zeros((E_batch.shape[0], 2))})
                         
    def on_epoch_end(self):
        np.random.shuffle(self.dataset['T_paired_ind'])
        np.random.shuffle(self.dataset['T_all_ind'])
        np.random.shuffle(self.dataset['E_all_ind'])
        return

    #---------------------------------------------------------
    def pop_batch(self):
        paired     = np.random.choice(self.dataset['T_paired_ind'],self.n_paired_per_batch,replace=False)
        unpaired_T = np.random.choice(self.dataset['T_all_ind'],self.batch_size - self.n_paired_per_batch,replace=False)
        unpaired_E = np.random.choice(self.dataset['E_all_ind'],self.batch_size - self.n_paired_per_batch,replace=False)

        T_batch  = np.concatenate((self.dataset['T'][paired,:], self.dataset['T'][unpaired_T,:]))
        E_batch  = np.concatenate((self.dataset['E'][paired,:], self.dataset['E'][unpaired_E,:]))
        ispaired = np.zeros((self.batch_size,1))
        ispaired[0:paired.size]=1
        
        return T_batch, E_batch, ispaired

