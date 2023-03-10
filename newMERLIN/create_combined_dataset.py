import os
import numpy as np
from mvalab import imz2mat


#import
TelecomParisZ4 = imz2mat('data\Saclay\TelecomParisZ4.IMA')[0]
TelecomParisOPT = imz2mat('data\Saclay\TelecomParisOPT.IMA')[0]

# check if folder exists
if not os.path.exists('dataset'):
    os.mkdir('dataset')
    os.mkdir('dataset/train')
    os.mkdir('dataset/validation')
    os.mkdir('dataset/train/spotlight')
    os.mkdir('dataset/validation/spotlight')

#training data
for i in range(TelecomParisZ4.shape[-1]):
    img_train_combined = np.zeros((1024,1024,5))
    img_train_combined[:,:,0] = np.real(TelecomParisZ4[:,:,i])
    img_train_combined[:,:,1] = np.imag(TelecomParisZ4[:,:,i])
    img_train_combined[:,:,2:5] = TelecomParisOPT
    np.save('dataset/train/spotlight/TelecomParisZ4OPT_'+str(i)+'.npy', img_train_combined)

#validation data
size=256
for i in range(TelecomParisZ4.shape[-1]):
    for j in range(1024//256):
        for k in range(1024//256):
            img_val_combined = np.zeros((256,256,5))
            img_val_combined[:,:,0] = np.real(TelecomParisZ4[:,:,i][k*size:(k+1)*size,j*size:(j+1)*size])
            img_val_combined[:,:,1] = np.imag(TelecomParisZ4[:,:,i][k*size:(k+1)*size,j*size:(j+1)*size])
            img_val_combined[:,:,2:5] = TelecomParisOPT[k*size:(k+1)*size,j*size:(j+1)*size,:]
            np.save('dataset/validation/spotlight/TelecomParisZ4_crop_'+str(i)+'_'+str(j)+'_'+str(k)+'.npy' ,img_val_combined)