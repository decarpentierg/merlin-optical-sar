# Navigate to appropriate directory
import os
while os.path.split(os.getcwd())[1] != "merlin-optical-sar":
    os.chdir("..")
    if os.getcwd() == "/":
        raise ValueError()
print("Current working directory: {}".format(os.getcwd()))
import numpy as np
import argparse

import sys
sys.path.insert(0, './utils')
import mvalab
import utils

parser = argparse.ArgumentParser(description='Extraction des données pour MERLIN')
parser.add_argument('--dataset', dest='dataset', default='Saclay', help='dataset to use : Saclay or Sendai')
parser.add_argument('--method', dest='method', default='SAR', help='method to use : SAR, SAR+OPT, SAR+SAR or SAR+OPT+SAR')
args = parser.parse_args()

if args.dataset != 'Saclay' and args.dataset != 'Sendai':
    raise ValueError('Dataset must be either Saclay or Sendai')
if args.method != 'SAR' and args.method != 'SAR+OPT' and args.method != 'SAR+SAR' and args.method != 'SAR+OPT+SAR':
    raise ValueError('Method must be either SAR, SAR+OPT, SAR+SAR or SAR+OPT+SAR')

#import
if args.dataset == 'Saclay':
    sar_no_register = mvalab.imz2mat('data/Saclay/TelecomParisZ4.IMA')[0]
    opt_no_register = mvalab.imz2mat('data/Saclay/TelecomParisOPT.IMA')[0][:, :, 0] #only the first image is relevant
    di, dj = -19, 147  # define registering translation
else:
    opt_no_register = mvalab.imz2mat('data/Sendai/PileTSX_AVANT_surTSX_1024x1024RECALOPT.IMA')[0][:988, 33:]
    sar_no_register = mvalab.imz2mat('data/Sendai/PileTSX_AVANT_surTSX_1024x1024RECALZ4.IMA')[0][:988, 33:,:]
    di, dj = 5, -23  # define registering translation

# register images
opt0 = np.copy(opt_no_register)
l,L = utils.register(opt0, sar_no_register[:, :, 0], di, dj)[0].shape
sar = np.zeros((l,L,sar_no_register.shape[-1]), dtype=complex)
for i in range(sar_no_register.shape[-1]):
    opt, sar[:, :, i]  = utils.register(opt0, sar_no_register[:, :, i],  di, dj)

# check if folder exists
folder_name = 'dataset_'+args.dataset+'_'+args.method
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
    os.mkdir(folder_name+'/train')
    os.mkdir(folder_name+'/validation')
    os.mkdir(folder_name+'/train/spotlight')
    os.mkdir(folder_name+'/validation/spotlight')

if args.method == 'SAR':
    nb_dim_sup = 0
elif args.method == 'SAR+OPT':
    nb_dim_sup = 1
elif args.method == 'SAR+SAR':
    nb_dim_sup = 1
elif args.method == 'SAR+OPT+SAR':
    nb_dim_sup = 2

#training data
for i in range(sar.shape[-1]):
    img_train_combined = np.zeros((l,L,2+nb_dim_sup))
    img_train_combined[:,:,0] = np.real(sar[:,:,i])
    img_train_combined[:,:,1] = np.imag(sar[:,:,i])
    if args.method == 'SAR+OPT':
        img_train_combined[:,:,2] = opt
    elif args.method == 'SAR+SAR':
        if args.dataset == 'Saclay':
            img_train_combined[:,:,2] = np.abs(sar[:,:,3]) #higher resolution SAR image
            if i >=2: # in this case we only use the first three sar images
                break
        else:
            img_train_combined[:,:,2] = np.abs(sar[:,:,1])
            break # only two sar images in Sendai
    elif args.method == 'SAR+OPT+SAR':
        img_train_combined[:,:,2] = opt
        if args.dataset == 'Saclay':
            img_train_combined[:,:,3] = np.abs(sar[:,:,3])
            if i >=2:
                break
        else:
            img_train_combined[:,:,3] = np.abs(sar[:,:,1])
            break
    np.save(folder_name+'/train/spotlight/train_data_'+str(i)+'.npy', img_train_combined)

print('Training data saved in '+folder_name+'/train/spotlight/')

#validation data
size=256
for i in range(sar.shape[-1]):
    for j in range(l//256):
        for k in range(L//256):
            img_val_combined = np.zeros((256,256,2+nb_dim_sup))
            img_val_combined[:,:,0] = np.real(sar[:,:,i])[k*size:(k+1)*size,j*size:(j+1)*size]
            img_val_combined[:,:,1] = np.imag(sar[:,:,i])[k*size:(k+1)*size,j*size:(j+1)*size]
            if args.method == 'SAR+OPT':
                img_val_combined[:,:,2] = opt[k*size:(k+1)*size,j*size:(j+1)*size]
            elif args.method == 'SAR+SAR':
                if args.dataset == 'Saclay':
                    img_val_combined[:,:,2] = np.abs(sar[:,:,3][k*size:(k+1)*size,j*size:(j+1)*size]) #higher resolution SAR image
                    if i >=2: # in this case we only use the first three sar images
                        break
                else:
                    img_val_combined[:,:,2] = np.abs(sar[:,:,1][k*size:(k+1)*size,j*size:(j+1)*size])
                    break # only two sar images in Sendai
            elif args.method == 'SAR+OPT+SAR':
                img_val_combined[:,:,2] = opt[k*size:(k+1)*size,j*size:(j+1)*size]
                if args.dataset == 'Saclay':
                    img_val_combined[:,:,3] = np.abs(sar[:,:,3][k*size:(k+1)*size,j*size:(j+1)*size]) 
                    if i >=2:
                        break
                else:
                    img_val_combined[:,:,3] = np.abs(sar[:,:,1][k*size:(k+1)*size,j*size:(j+1)*size])
                    break
            np.save(folder_name+'/validation/spotlight/val_data_'+str(i)+'_'+str(j)+'_'+str(k)+'.npy', img_val_combined)

print('Validation data saved in '+folder_name+'/validation/spotlight/')