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
from tensorflow.python.keras.callbacks import (Callback, CSVLogger,
                                               ModelCheckpoint)
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import (BatchNormalization, Conv2D, Dense,
                                            Dropout, Flatten, Input, Lambda,
                                            MaxPooling2D, Reshape,
                                            UpSampling2D)
from tensorflow.python.keras.models import Model
from coupling_functions import fullcov, minvar, mse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",    default=150,     type=int,   help="Batch size")
parser.add_argument("--p_drop",        default=0.4,     type=float, help="Dropout rate")
parser.add_argument("--latent_dim",    default=2,       type=int,   help="Number of latent dims")

parser.add_argument("--cpl_fn",        default="minvar",type=str,   help="mse, mseBN, fullcov or minvar")
parser.add_argument("--cpl_str",       default=1e-3,    type=float, help="coupling strength")

parser.add_argument("--n_epoch",       default=500,     type=int,   help="n(training epochs)")
parser.add_argument("--exp_name",      default='MNIST', type=str,   help="Folder name to store results")
parser.add_argument("--model_id",      default='cnn',   type=str,   help="Model id part of result filenames")
parser.add_argument("--run_iter",      default=0,       type=int,   help="Run-specific id")


def main(batch_size=150, p_drop=0.4, latent_dim=2,
         cpl_fn='minvar', cpl_str=1e-3,
         n_epoch=500, run_iter=0, model_id='cnn',exp_name='MNIST'):
    
    
    fileid = model_id + \
        '_cf_' + cpl_fn + \
        '_cs_' + str(cpl_str) + \
        '_pd_' + str(p_drop) + \
        '_bs_' + str(batch_size) + \
        '_ld_' + str(latent_dim) + \
        '_ne_' + str(n_epoch) + \
        '_ri_' + str(run_iter)

    fileid = fileid.replace('.', '-')
    train_dat, train_lbl, val_dat, val_lbl, dir_pth = dataIO(exp_name=exp_name)
    
    #Architecture parameters ------------------------------
    input_dim = train_dat.shape[1]
    n_arms = 2
    fc_dim = 49
    
    #Model definition -------------------------------------
    M = {}
    M['in_ae'] = Input(shape=(28,28,1), name='in_ae')
    for i in range(n_arms):
        M['co1_ae_'+str(i)]  = Conv2D(10, (3, 3), activation='relu', padding='same',name='co1_ae_'+str(i))(M['in_ae'])
        M['mp1_ae_'+str(i)]  = MaxPooling2D((2, 2), padding='same',name='mp1_ae_'+str(i))(M['co1_ae_'+str(i)])
        M['dr1_ae_'+str(i)]  = Dropout(rate=p_drop, name='dr1_ae_'+str(i))(M['mp1_ae_'+str(i)])
        M['fl1_ae_'+str(i)]  = Flatten(name='fl1_ae_'+str(i))(M['dr1_ae_'+str(i)])
        M['fc01_ae_'+str(i)] = Dense(fc_dim, activation='relu', name='fc01_ae_'+str(i))(M['fl1_ae_'+str(i)])
        M['fc02_ae_'+str(i)] = Dense(fc_dim, activation='relu', name='fc02_ae_'+str(i))(M['fc01_ae_'+str(i)])
        M['fc03_ae_'+str(i)] = Dense(fc_dim, activation='relu', name='fc03_ae_'+str(i))(M['fc02_ae_'+str(i)])

        if cpl_fn in ['mse']:
            M['ld_ae_'+str(i)] = Dense(latent_dim, activation='linear', name='ld_ae_'+str(i))(M['fc03_ae_'+str(i)])
        elif cpl_fn in ['mseBN', 'fullcov', 'minvar']:
            M['fc04_ae_'+str(i)] = Dense(latent_dim, activation='linear', name='fc04_ae_'+str(i))(M['fc03_ae_'+str(i)])
            M['ld_ae_'+str(i)] = BatchNormalization(scale=False, center=False, epsilon=1e-10, momentum=0.99, name='ld_ae_'+str(i))(M['fc04_ae_'+str(i)])

        M['fc05_ae_'+str(i)] = Dense(fc_dim, activation='relu', name='fc05_ae_'+str(i))(M['ld_ae_'+str(i)])
        M['fc06_ae_'+str(i)] = Dense(fc_dim, activation='relu', name='fc06_ae_'+str(i))(M['fc05_ae_'+str(i)])
        M['fc07_ae_'+str(i)] = Dense(fc_dim*4, activation='relu', name='fc07_ae_'+str(i))(M['fc06_ae_'+str(i)])
        M['re1_ae_'+str(i)]  = Reshape((14, 14, 1), name='re1_ae_'+str(i))(M['fc07_ae_'+str(i)])
        M['us1_ae_'+str(i)]  = UpSampling2D((2, 2),name = 'us1_ae_'+str(i))(M['re1_ae_'+str(i)])
        M['co2_ae_'+str(i)]  = Conv2D(10, (3, 3), activation='relu', padding='same',name='co2_ae_'+str(i))(M['us1_ae_'+str(i)])
        M['ou_ae_'+str(i)]   = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name='ou_ae_'+str(i))(M['co2_ae_'+str(i)])

    cplAE = Model(inputs=M['in_ae'],
                  outputs=[M['ou_ae_'+str(i)] for i in range(n_arms)] + [M['ld_ae_'+str(i)] for i in range(n_arms)])
                           
    if cpl_fn in ['mse','mseBN']:
        cpl_fn_loss = mse
    elif cpl_fn == 'fullcov':
        cpl_fn_loss = fullcov
    elif cpl_fn == 'minvar':
        cpl_fn_loss = minvar

    assert type(cpl_fn)
    #Create loss dictionary
    loss_dict = {'ou_ae_0': mse(M['in_ae'],M['ou_ae_0']), 
                 'ou_ae_1': mse(M['in_ae'],M['ou_ae_1']),
                 'ld_ae_0': cpl_fn_loss(M['ld_ae_0'], M['ld_ae_1']),
                 'ld_ae_1': cpl_fn_loss(M['ld_ae_1'], M['ld_ae_0'])}

    
    #Loss weights dictionary
    loss_wt_dict = {'ou_ae_0': 1.0, 'ou_ae_1': 1.0,
                    'ld_ae_0': cpl_str, 
                    'ld_ae_1': cpl_str}

    #Add loss definitions to the model
    cplAE.compile(optimizer='adam', loss=loss_dict, loss_weights=loss_wt_dict)
    
    #Data feed
    train_input_dict = {'in_ae': train_dat}
    val_input_dict   = {'in_ae': val_dat}
    train_output_dict = {'ou_ae_0': train_dat, 
                         'ou_ae_1': train_dat, 
                         'ld_ae_0': np.empty((train_dat.shape[0], latent_dim)), 
                         'ld_ae_1': np.empty((train_dat.shape[0], latent_dim))}
    val_output_dict = {'ou_ae_0': val_dat, 
                       'ou_ae_1': val_dat, 
                       'ld_ae_0': np.empty((val_dat.shape[0], latent_dim)), 
                       'ld_ae_1': np.empty((val_dat.shape[0], latent_dim))}
    
    log_cb = CSVLogger(filename=dir_pth['logs']+fileid+'.csv')
    
    #Train model
    cplAE.fit(train_input_dict, train_output_dict,
                validation_data=(val_input_dict, val_output_dict),
                batch_size=batch_size, initial_epoch=0, epochs=n_epoch,
                verbose=2, shuffle=True,
                callbacks = [log_cb])
 
    #Saving weights
    cplAE.save_weights(dir_pth['result']+fileid+'-modelweights'+'.h5')

    matsummary = {}
    #Trained model prediction
    for i in range(n_arms):
        encoder = Model(inputs=M['in_ae'], outputs=M['ld_ae_'+str(i)])
        matsummary['z_val_'+str(i)] = encoder.predict({'in_ae': val_dat})
        matsummary['z_train_'+str(i)] = encoder.predict({'in_ae': train_dat})
    matsummary['train_lbl']=train_lbl
    matsummary['val_lbl']=val_lbl
    sio.savemat(dir_pth['result']+fileid+'-summary.mat', matsummary)
    return

def dataIO(exp_name='MNIST'):
    from pathlib import Path

    dir_pth = {}
    curr_path = str(Path().absolute()) + '/'
    dir_pth['data'] = curr_path + 'data/raw/'
    dir_pth['result'] = curr_path + 'data/results/' + exp_name + '/'
    dir_pth['logs'] = dir_pth['result'] + 'logs/'
    Path(dir_pth['logs']).mkdir(parents=True, exist_ok=True) 

    (train_dat, train_lbl), (val_dat, val_lbl) = mnist.load_data()
    
    train_dat = np.reshape(train_dat, (len(train_dat), 28, 28, 1))
    val_dat = np.reshape(val_dat, (len(val_dat), 28, 28, 1))
    train_dat = train_dat.astype('float32') / 255.
    val_dat = val_dat.astype('float32') / 255.
    return train_dat, train_lbl, val_dat, val_lbl, dir_pth

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
