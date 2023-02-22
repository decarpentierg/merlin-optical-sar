import glob
import random
import os
import numpy as np
from scipy import signal
from scipy import special


'''
Generate patches for the images in the folder dataset/data/Train
The code scans among the training images and then for data_aug_times
'''

class GenerateDataset():

    def generate_patches(self,src_dir="./dataset/data/Train",pat_size=256,step=0,stride=64,bat_size=4,data_aug_times=1,n_channels=2):
        count = 0
        filepaths = glob.glob(src_dir + '/*.npy')
        print("number of training data %d" % len(filepaths))

        # calculate the number of patches
        for i in range(len(filepaths)):
            img = np.load(filepaths[i])

            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0+step, (im_h - pat_size), stride):
                for y in range(0+step, (im_w - pat_size), stride):
                    count += 1
        origin_patch_num = count * data_aug_times

        if origin_patch_num % bat_size != 0:
            numPatches = (origin_patch_num / bat_size + 1) * bat_size
        else:
            numPatches = origin_patch_num
        print("total patches = %d , batch size = %d, total batches = %d" % \
              (numPatches, bat_size, numPatches / bat_size))

        # data matrix 4-D
        numPatches=int(numPatches)
        inputs = np.zeros((numPatches, pat_size, pat_size, n_channels), dtype="float32")


        count = 0
        # generate patches
        for i in range(len(filepaths)): #scan through images
            img = np.load(filepaths[i])
            img_s = img
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            for x in range(0 + step, im_h - pat_size, stride):
                for y in range(0 + step, im_w - pat_size, stride):
                    inputs[count, :, :, :] = img_s[x:x + pat_size, y:y + pat_size, :2]
                    count += 1


        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

        return inputs
