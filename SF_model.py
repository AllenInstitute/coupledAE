import argparse
import os
import pdb
import re
import socket
import sys
import timeit

import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from keras.layers import BatchNormalization, Dense, Dropout, Input, Lambda
from keras.losses import mean_squared_error as mse
from keras.models import Model,load_model
from datagen import DatagenTE, dataset_50foldSF

parser = argparse.ArgumentParser()

parser.add_argument("--ngenes",            default=5000,                 type=int,   help="Number of genes")
parser.add_argument("--batch_size",        default=500,                  type=int,   help="Coupling strength")
parser.add_argument("--n_paired_per_batch",default=200,                  type=int,   help="Number of paired examples")
parser.add_argument("--cvset",             default=0,                    type=int,   help="50-fold cross validation set number")

parser.add_argument("--p_drop",            default=0.5,                  type=float, help="Dropout rate T arm")
parser.add_argument("--latent_dim",        default=2,                    type=int,   help="Number of latent dims")

parser.add_argument("--cpl_str",           default=10.0,                 type=float, help="Coupling strength")

parser.add_argument("--n_epoch",           default=5000,                 type=int,   help="Number of epochs to train")
parser.add_argument("--steps_per_epoch",   default=20,                   type=int,   help="Number of gradient steps per epoch")
parser.add_argument("--warm_start",        default=True,                 type=bool,  help="Start from pre-trained single autoencoder weights")

parser.add_argument("--run_iter",          default=0,                    type=int,   help="Run-specific id")
parser.add_argument("--modelid",           default='SF',               type=str,   help="Model-specific id")
parser.add_argument("--exp_name",          default='split-facs-5kgenes', type=str,   help="Experiment set")
parser.add_argument("--dname",             default='dat2_4k',            type=str,   help="Dataset")

def main(ngenes=5000, batch_size=500, n_paired_per_batch=200, cvset=0, 
         p_drop=0.5,latent_dim=2,cpl_str=10.0,
         n_epoch=5000, steps_per_epoch = 20, warm_start=True,
         run_iter=0, modelid='SF',exp_name='split-facs',dname='dat2_4k'):
              
    train_dat, val_dat, train_ind_T, train_ind_E, val_ind_T, val_ind_E, dir_pth = dataset_50foldSF(exp_name=exp_name,fname=dname,ngenes=ngenes,val_perc=0.1)

    train_generator = DatagenTE(dataset=train_dat, batch_size=batch_size, n_paired_per_batch=n_paired_per_batch, steps_per_epoch=steps_per_epoch)
    chkpt_save_period = 1e3
    
    #Architecture parameters ------------------------------------------------------------
    input_dim  = [train_dat['T'].shape[1],train_dat['E'].shape[1]]

    fileid = modelid + \
        '_ng_'  + str(ngenes) + \
        '_cs_'  + str(cpl_str) + \
        '_pd_' +  str(p_drop) + \
        '_bs_'  + str(batch_size) + \
        '_np_'  + str(n_paired_per_batch) + \
        '_se_'  + str(steps_per_epoch) +\
        '_ne_'  + str(n_epoch) + \
        '_cv_'  + str(cvset) + \
        '_ri_'  + str(run_iter) + \
        '_ws_'  + str(warm_start) + \
        '_dn_'  + str(dname)
    fileid = fileid.replace('.', '-')
    print(fileid)
    

    fc_dimT=[100,100,100,100]
    #Model inputs -----------------------------------------------------------------------
    M = {}
    M['in_ae_0']   = Input(shape=(input_dim[0],), name='in_ae_0')
    M['in_ae_1']   = Input(shape=(input_dim[1],), name='in_ae_1')

    M['ispaired_ae_0']   = Input(shape=(1,), name='ispaired_ae_0')
    M['ispaired_ae_1']   = Input(shape=(1,), name='ispaired_ae_1')

    #T arm-------------------------------------------------------------------------------
    M['dr_ae_0'] = Dropout(p_drop, name='dr_ae_0')(M['in_ae_0'])
    X = 'dr_ae_0'

    for j, units in enumerate(fc_dimT):
        Y = 'fc'+ format(j,'02d') +'_ae_0'
        M[Y] = Dense(units, activation='relu', name=Y)(M[X])
        X = Y

    M['ldx_ae_0'] = Dense(latent_dim, activation='linear',name='ldx_ae_0')(M[X])
    M['ld_ae_0']   = BatchNormalization(scale = False, center = False ,epsilon = 1e-10, momentum = 0.99, name='ld_ae_0')(M['ldx_ae_0'])
    X = 'ld_ae_0'

    for j, units in enumerate(reversed(fc_dimT)):
        Y = 'fc'+ format(j+len(fc_dimT),'02d') +'_ae_0'
        M[Y] = Dense(units, activation='relu', name=Y)(M[X])
        X = Y
    
    M['ou_ae_0']   = Dense(input_dim[0], activation='relu', name='ou_ae_0')(M[X])

    #E arm-------------------------------------------------------------------------------
    M['dr_ae_1'] = Dropout(p_drop, name='dr_ae_1')(M['in_ae_1'])
    X = 'dr_ae_1'

    for j, units in enumerate(fc_dimT):
        Y = 'fc'+ format(j,'02d') +'_ae_1'
        M[Y] = Dense(units, activation='relu', name=Y)(M[X])
        X = Y

    M['ldx_ae_1'] = Dense(latent_dim, activation='linear',name='ldx_ae_1')(M[X])
    M['ld_ae_1']   = BatchNormalization(scale = False, center = False ,epsilon = 1e-10, momentum = 0.99, name='ld_ae_1')(M['ldx_ae_1'])
    X = 'ld_ae_1'

    for j, units in enumerate(reversed(fc_dimT)):
        Y = 'fc'+ format(j+len(fc_dimT),'02d') +'_ae_1'
        M[Y] = Dense(units, activation='relu', name=Y)(M[X])
        X = Y
    
    M['ou_ae_1']   = Dense(input_dim[0], activation='relu', name='ou_ae_1')(M[X])

    cplAE = Model(inputs=[M['in_ae_0'], M['in_ae_1'], M['ispaired_ae_0'], M['ispaired_ae_1']],
                  outputs=[M['ou_ae_0'], M['ou_ae_1'],M['ld_ae_0'], M['ld_ae_1']])
    
    def coupling_loss(zi, pairedi, zj, pairedj):
        '''Minimum eigenvalue based loss. Analogous to full covariance based loss.
        \n SVD is calculated over all datapoints
        \n MSE is calculated over only `paired` datapoints'''
        batch_size = tf.shape(zi)[0]

        paired_i = tf.reshape(pairedi, [tf.shape(pairedi)[0],])
        paired_j = tf.reshape(pairedj, [tf.shape(pairedj)[0],])
        zi_paired = tf.boolean_mask(zi, tf.equal(paired_i, 1.0))
        zj_paired = tf.boolean_mask(zj, tf.equal(paired_j, 1.0))

        vars_j_ = tf.square(tf.svd(zj - tf.reduce_mean(zj, axis=0), compute_uv=False))/tf.cast(batch_size - 1, tf.float32)
        vars_j  = tf.where(tf.is_nan(vars_j_), tf.zeros_like(vars_j_) + tf.cast(1e-1,dtype=tf.float32), vars_j_)
        L_ij    = tf.losses.mean_squared_error(zi_paired, zj_paired)/tf.maximum(tf.reduce_min(vars_j, axis=None),tf.cast(1e-2,dtype=tf.float32))

        def loss(y_true, y_pred):
            #Adaptive version:#tf.multiply(tf.stop_gradient(L_ij), L_ij)
            return L_ij
        return loss
        
    #Create loss dictionary
    loss_dict = {'ou_ae_0': mse, 'ou_ae_1': mse,
                 'ld_ae_0': coupling_loss(zi=M['ld_ae_0'], pairedi=M['ispaired_ae_0'],zj=M['ld_ae_1'], pairedj=M['ispaired_ae_1']),
                 'ld_ae_1': coupling_loss(zi=M['ld_ae_1'], pairedi=M['ispaired_ae_1'],zj=M['ld_ae_0'], pairedj=M['ispaired_ae_0'])}

    #Loss weights dictionary
    loss_wt_dict = {'ou_ae_0': 1.0,
                    'ou_ae_1': 1.0,
                    'ld_ae_0': cpl_str,
                    'ld_ae_1': cpl_str}

    #Add loss definitions to the model
    cplAE.compile(optimizer='adam', loss=loss_dict, loss_weights=loss_wt_dict)

    #Checkpoint function definitions
    checkpoint_cb = ModelCheckpoint(filepath=(dir_pth['checkpoint']+fileid + '-checkpoint-' + '{epoch:04d}' + '.h5'),
                                      verbose=1, save_best_only=False, save_weights_only=True,
                                      mode='auto', period=chkpt_save_period)

    minind = np.min([val_dat['T'].shape[0],val_dat['E'].shape[0]])
    val_in = {'in_ae_0': val_dat['T'][:minind,:],
              'in_ae_1': val_dat['E'][:minind,:],
              'ispaired_ae_0': val_dat['T_ispaired'][:minind],
              'ispaired_ae_1': val_dat['E_ispaired'][:minind]}

    val_out = {'ou_ae_0': val_dat['T'][:minind,:],
               'ou_ae_1': val_dat['E'][:minind,:],
               'ld_ae_0': np.zeros((minind, latent_dim)),
               'ld_ae_1': np.zeros((minind, latent_dim))}
    
    #Custom callback object
    log_cb = CSVLogger(filename=dir_pth['logs']+fileid+'.csv')

    #Warm start
    if warm_start:
        print('Searching for pretrained model: '+dir_pth['result'] + 'warm_start-model.h5')
        pre_trained = load_model(dir_pth['result'] + 'warm_start-model.h5')
        for cplAE_layer in cplAE.layers:
            for pt_layer in pre_trained.layers:
                if pt_layer.name[:-1]==cplAE_layer.name[:-1]:
                    #print(cplAE_layer.name +' weights set from '+ pt_layer.name[:-1])
                    cplAE_layer.set_weights(pt_layer.get_weights())
        
    #validation_data=(val_in,val_out),
    last_checkpoint_epoch = 0
    start_time = timeit.default_timer()
    cplAE.fit_generator(train_generator,
                        epochs=n_epoch,
                        max_queue_size=100,
                        use_multiprocessing=False, workers=1,
                        initial_epoch=last_checkpoint_epoch,
                        verbose=2, callbacks=[checkpoint_cb,log_cb])

    
    elapsed = timeit.default_timer() - start_time        
    print('-------------------------------')
    print('Training time:',elapsed)
    print('-------------------------------')

    #Saving weights
    cplAE.save_weights(dir_pth['result']+fileid+'-modelweights'+'.h5')
    
    matsummary = {}
    matsummary['cvset']       = cvset
    matsummary['val_ind_T']   = val_ind_T
    matsummary['val_ind_E']   = val_ind_E
    matsummary['train_ind_T'] = train_ind_T
    matsummary['train_ind_E'] = train_ind_E
    
    #Trained model predictions
    i = 0
    encoder = Model(inputs=M['in_ae_'+str(i)], outputs=M['ld_ae_'+str(i)])
    matsummary['z_val_'+str(i)]   = encoder.predict({'in_ae_'+str(i): val_dat['T']})
    matsummary['z_train_'+str(i)] = encoder.predict({'in_ae_'+str(i): train_dat['T']})

    i = 1
    encoder = Model(inputs=M['in_ae_'+str(i)], outputs=M['ld_ae_'+str(i)])
    matsummary['z_val_'+str(i)]   = encoder.predict({'in_ae_'+str(i): val_dat['E']})
    matsummary['z_train_'+str(i)] = encoder.predict({'in_ae_'+str(i): train_dat['E']})

    sio.savemat(dir_pth['result']+fileid+'-summary', matsummary)
    print('saved files')
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))