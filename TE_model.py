# -----------------------------------------------
# 5341 exclusive, 3585 matched, total 8926 in T
# -----------------------------------------------
# 0 exclusive, 3585 matched, total 3585 in E

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
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout, Input, Lambda
from tensorflow.python.keras.losses import mean_squared_error as mse
from tensorflow.python.keras.models import Model
from datagen import DatagenTE, dataset_50fold

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",        default=100,                  type=int,     help="Coupling strength")
parser.add_argument("--n_paired_per_batch",default=100,                  type=int,     help="Number of paired examples")
parser.add_argument("--cvset"             ,default=0,                    type=int,     help="50-fold cross validation set number")

parser.add_argument("--p_dropT",           default=0.5,                  type=float,   help="Dropout rate T arm")
parser.add_argument("--p_dropE",           default=0.1,                  type=float,   help="Dropout rate E arm")
parser.add_argument("--stdE",              default=0.05,                 type=float,   help="Gaussian noise sigma E arm")

parser.add_argument("--fc_dimT",           default=[50,50,50,50],        type=int,     help="List of dims for T fc layers", nargs = '+')
parser.add_argument("--fc_dimE",           default=[60,60,60,60],        type=int,     help="List of dims for E fc layers", nargs = '+')
parser.add_argument("--latent_dim",        default=3,                    type=int,     help="Number of latent dims")

parser.add_argument("--recon_strT",        default=1.0,                  type=float,   help="Reconstruction strength T arm")
parser.add_argument("--recon_strE",        default=0.1,                  type=float,   help="Reconstruction strength E arm")
parser.add_argument("--cpl_str",           default=10.0,                 type=float,   help="Coupling strength")

parser.add_argument("--n_epoch",           default=2000,                 type=int,     help="Number of epochs to train")
parser.add_argument("--steps_per_epoch",   default=500,                  type=int,     help="Number of gradient steps per epoch")

parser.add_argument("--run_iter",          default=0,                    type=int,     help="Run-specific id")
parser.add_argument("--model_id",          default='crossval',           type=str,     help="Model-specific id")
parser.add_argument("--exp_name",          default='patchseq_v2_noadapt',type=str,     help="Experiment set")


def main(batch_size=100, n_paired_per_batch=100, cvset=0, 
         p_dropT=0.5, p_dropE=0.1, stdE=0.05,
         fc_dimT=[50,50,50,50],fc_dimE=[60,60,60,60],latent_dim=3,
         recon_strT=1.0, recon_strE=0.1, cpl_str=10.0,
         n_epoch=2000, steps_per_epoch = 500, 
         run_iter=0, model_id='crossval_noadaptloss',exp_name='patchseq_v2_noadapt'):
         
    train_dat, val_dat, train_ind_T, train_ind_E, val_ind, dir_pth = dataset_50fold(exp_name=exp_name,cvset=cvset)
    train_generator = DatagenTE(dataset=train_dat, batch_size=batch_size, n_paired_per_batch=n_paired_per_batch, steps_per_epoch=steps_per_epoch)
    chkpt_save_period = 1e7
    
    #Architecture parameters ------------------------------
    input_dim  = [train_dat['T'].shape[1],train_dat['E'].shape[1]]

    #'_fcT_' +  '-'.join(map(str, fc_dimT)) + \
    #'_fcE_' +  '-'.join(map(str, fc_dimE)) + \
    fileid = model_id + \
        '_rT_' + str(recon_strT) + \
        '_rE_'  + str(recon_strE) + \
        '_cs_'  + str(cpl_str) + \
        '_pdT_' + str(p_dropT) + \
        '_pdE_' + str(p_dropE) + \
        '_sdE_' + str(stdE) + \
        '_bs_'  + str(batch_size) + \
        '_np_'  + str(n_paired_per_batch) + \
        '_se_'  + str(steps_per_epoch) +\
        '_ne_'  + str(n_epoch) + \
        '_cv_'  + str(cvset) + \
        '_ri_'  + str(run_iter)
    fileid = fileid.replace('.', '-')
    
    print(fileid)
    out_actfcn = ['elu','linear']

    def add_gauss_noise(x):
        '''Injects additive gaussian noise independently into each element of input x'''
        x_noisy = x + tf.random.normal(shape=tf.shape(x), mean=0., stddev=stdE, dtype = tf.float32)
        return tf.keras.backend.in_train_phase(x_noisy, x)
    
    #Model inputs -----------------------------------------
    M = {}
    M['in_ae_0']   = Input(shape=(input_dim[0],), name='in_ae_0')
    M['in_ae_1']   = Input(shape=(input_dim[1],), name='in_ae_1')

    M['ispaired_ae_0'] = Input(shape=(1,), name='ispaired_ae_0')
    M['ispaired_ae_1'] = Input(shape=(1,), name='ispaired_ae_1')

    #Transcriptomics arm---------------------------------------------------------------------------------
    M['dr_ae_0'] = Dropout(p_dropT, name='dr_ae_0')(M['in_ae_0'])
    X = 'dr_ae_0'

    for j, units in enumerate(fc_dimT):
        Y = 'fc'+ format(j,'02d') +'_ae_0'
        M[Y] = Dense(units, activation='elu', name=Y)(M[X])
        X = Y

    M['ldx_ae_0'] = Dense(latent_dim, activation='linear',name='ldx_ae_0')(M[X])
    M['ld_ae_0']  = BatchNormalization(scale = False, center = False ,epsilon = 1e-10, momentum = 0.99, name='ld_ae_0')(M['ldx_ae_0'])
    X = 'ld_ae_0'

    for j, units in enumerate(reversed(fc_dimT)):
        Y = 'fc'+ format(j+len(fc_dimT),'02d') +'_ae_0'
        M[Y] = Dense(units, activation='elu', name=Y)(M[X])
        X = Y
    
    M['ou_ae_0']  = Dense(input_dim[0], activation=out_actfcn[0], name='ou_ae_0')(M[X])

    #Electrophysiology arm--------------------------------------------------------------------------------
    M['no_ae_1']  = Lambda(add_gauss_noise,name='no_ae_1')(M['in_ae_1'])
    M['dr_ae_1']  = Dropout(p_dropE, name='dr_ae_1')(M['no_ae_1'])
    X = 'dr_ae_1'
    for j, units in enumerate(fc_dimE):
        Y = 'fc'+ format(j,'02d') +'_ae_1'
        M[Y] = Dense(units, activation='elu', name=Y)(M[X])
        X = Y
    
    M['ldx_ae_1'] = Dense(latent_dim, activation='linear',name='ldx_ae_1')(M[X])
    M['ld_ae_1']  = BatchNormalization(scale = False, center = False ,epsilon = 1e-10, momentum = 0.99, name='ld_ae_1')(M['ldx_ae_1'])
    X = 'ld_ae_1'

    for j, units in enumerate(reversed(fc_dimE)):
        Y = 'fc'+ format(j+len(fc_dimE),'02d') +'_ae_1'
        M[Y] = Dense(units, activation='elu', name=Y)(M[X])
        X = Y

    M['ou_ae_1']  = Dense(input_dim[1], activation=out_actfcn[1], name='ou_ae_1')(M[X])

    cplAE = Model(inputs=[M['in_ae_0'], M['in_ae_1'], M['ispaired_ae_0'], M['ispaired_ae_1']],
                  outputs=[M['ou_ae_0'], M['ou_ae_1'],M['ld_ae_0'], M['ld_ae_1']])
    
    def coupling_loss(zi, pairedi, zj, pairedj):
        '''Minimum singular value based loss. 
        \n SVD is calculated over all datapoints
        \n MSE is calculated over only `paired` datapoints'''
        batch_size = tf.shape(zi)[0]

        paired_i = tf.reshape(pairedi, [tf.shape(pairedi)[0],])
        paired_j = tf.reshape(pairedj, [tf.shape(pairedj)[0],])
        zi_paired = tf.boolean_mask(zi, tf.equal(paired_i, 1.0))
        zj_paired = tf.boolean_mask(zj, tf.equal(paired_j, 1.0))

        vars_j_ = tf.square(tf.linalg.svd(zj - tf.reduce_mean(zj, axis=0), compute_uv=False))/tf.cast(batch_size - 1, tf.float32)
        vars_j  = tf.where(tf.math.is_nan(vars_j_), tf.zeros_like(vars_j_) + tf.cast(1e-1,dtype=tf.float32), vars_j_)
        L_ij    = tf.compat.v1.losses.mean_squared_error(zi_paired, zj_paired)/tf.maximum(tf.reduce_min(vars_j, axis=None),tf.cast(1e-2,dtype=tf.float32))

        def loss(y_true, y_pred):
            #Adaptive version:#tf.multiply(tf.stop_gradient(L_ij), L_ij)
            return L_ij
        return loss
        
    #Create loss dictionary
    loss_dict = {'ou_ae_0': mse, 'ou_ae_1': mse,
                 'ld_ae_0': coupling_loss(zi=M['ld_ae_0'], pairedi=M['ispaired_ae_0'],zj=M['ld_ae_1'], pairedj=M['ispaired_ae_1']),
                 'ld_ae_1': coupling_loss(zi=M['ld_ae_1'], pairedi=M['ispaired_ae_1'],zj=M['ld_ae_0'], pairedj=M['ispaired_ae_0'])}

    #Loss weights dictionary
    loss_wt_dict = {'ou_ae_0': recon_strT,
                    'ou_ae_1': recon_strE,
                    'ld_ae_0': cpl_str,
                    'ld_ae_1': cpl_str}

    #Add loss definitions to the model
    cplAE.compile(optimizer='adam', loss=loss_dict, loss_weights=loss_wt_dict)

    #Checkpoint function definitions
    checkpoint_cb = ModelCheckpoint(filepath=(dir_pth['checkpoint']+fileid + '-checkpoint-' + '{epoch:04d}' + '.h5'),
                                      verbose=1, save_best_only=False, save_weights_only=True,
                                      mode='auto', period=chkpt_save_period)

    val_in = {'in_ae_0': val_dat['T'],
              'in_ae_1': val_dat['E'],
              'ispaired_ae_0': val_dat['T_ispaired'],
              'ispaired_ae_1': val_dat['E_ispaired']}

    val_out = {'ou_ae_0': val_dat['T'],
               'ou_ae_1': val_dat['E'],
               'ld_ae_0': np.zeros((val_dat['T'].shape[0], latent_dim)),
               'ld_ae_1': np.zeros((val_dat['E'].shape[0], latent_dim))}
    
    #Custom callback object
    log_cb = CSVLogger(filename=dir_pth['logs']+fileid+'.csv')

    last_checkpoint_epoch = 0
    start_time = timeit.default_timer()
    cplAE.fit_generator(train_generator,
                        validation_data=(val_in,val_out),
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
    matsummary['val_ind']     = val_ind
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
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))