import time
import numpy as np
import os

from utils import *
from scipy import special
import argparse


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601
L = 1
c = (1 / 2) * (special.psi(L) - np.log(L))
cn = c / (M - m)  # normalized (0,1) mean of log speckle

import torch
import numpy as np



class AE(torch.nn.Module):

    def __init__(self,batch_size,eval_batch_size,device,method):
        super().__init__()

        self.batch_size=batch_size
        self.eval_batch_size=eval_batch_size
        self.device=device
        self.method=method

        self.x = None
        self.height = None
        self.width = None
        self.out_channels = None
        self.kernel_size_cv2d = None
        self.stride_cv2d = None
        self.padding_cv2d = None
        self.kernel_size_mp2d = None
        self.stride_mp2d = None
        self.padding_mp2d = None
        self.alpha = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky = torch.nn.LeakyReLU(0.1)

        self.enc0 = torch.nn.Conv2d(in_channels=1, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        
        # add layers to process the additional images
        ############################################################################################################

        if method != 'SAR':
          if method == 'SAR+OPT' or method == 'SAR+SAR' :
            n_channels_sup = 1
          else:
            n_channels_sup = 2
          self.enc0_opt = torch.nn.Conv2d(in_channels=n_channels_sup, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                        padding='same')
          
          self.enc1_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same')
          self.enc2_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same')
          self.enc3_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same')
          self.enc4_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same')
          self.enc5_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same')
          self.enc6_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same')
          
        ############################################################################################################
              

        self.dec5 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1a = torch.nn.Conv2d(in_channels=97, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self,x ,batch_size):
        """  Defines a class for an autoencoder algorithm for an object (image) x

        An autoencoder is a specific type of feedforward neural networks where the
        input is the same as the
        output. It compresses the input into a lower-dimensional code and then
        reconstruct the output from this representattion. It is a dimensionality
        reduction algorithm

        Parameters
        ----------
        x : np.array
        a numpy array containing image

        Returns
        ----------
        x-n : np.array
        a numpy array containing the denoised image i.e the image itself minus the noise

        """
        x_SAR=torch.reshape(x[:,:,:,:,0], [batch_size, 1, 256, 256])
        skips_SAR = [x_SAR]

        n_SAR = x_SAR

        # ENCODER
        n_SAR = self.leaky(self.enc0(n_SAR))
        n_SAR = self.leaky(self.enc1(n_SAR))
        n_SAR = self.pool(n_SAR)
        skips_SAR.append(n_SAR)

        n_SAR = self.leaky(self.enc2(n_SAR))
        n_SAR = self.pool(n_SAR)
        skips_SAR.append(n_SAR)

        n_SAR = self.leaky(self.enc3(n_SAR))
        n_SAR = self.pool(n_SAR)
        skips_SAR.append(n_SAR)

        n_SAR = self.leaky(self.enc4(n_SAR))
        n_SAR = self.pool(n_SAR)
        skips_SAR.append(n_SAR)

        n_SAR = self.leaky(self.enc5(n_SAR))
        n_SAR = self.pool(n_SAR)
        n_SAR = self.leaky(self.enc6(n_SAR))

        ############################################################################################################
        # ENCODER OPTIC 
        if self.method != 'SAR':
          if self.method == 'SAR+OPT' or self.method == 'SAR+SAR' :
            n_channels_sup = 1
          else:
            n_channels_sup = 2
          x_sup = torch.reshape(x[:, :, :,:, 1:], [batch_size, n_channels_sup, 256, 256])
          skips_sup = [x_sup]

          n_sup = x_sup

          n_sup = self.leaky(self.enc0_opt(n_sup))
          n_sup = self.leaky(self.enc1_opt(n_sup))
          n_sup = self.pool(n_sup)
          skips_sup.append(n_sup)

          n_sup = self.leaky(self.enc2_opt(n_sup))
          n_sup = self.pool(n_sup)
          skips_sup.append(n_sup)

          n_sup = self.leaky(self.enc3_opt(n_sup))
          n_sup = self.pool(n_sup)
          skips_sup.append(n_sup)

          n_sup = self.leaky(self.enc4_opt(n_sup))
          n_sup = self.pool(n_sup)
          skips_sup.append(n_sup)

          n_sup = self.leaky(self.enc5_opt(n_sup))
          n_sup = self.pool(n_sup)
          n_sup = self.leaky(self.enc6_opt(n_sup))

          # COMBINE
          n = n_SAR + n_sup
        else:
          n = n_SAR


        ############################################################################################################


        # DECODER
        n = self.upscale2d(n)
        if self.method != 'SAR':
          skips = skips_SAR.pop()+ skips_sup.pop()
        else:
          skips = skips_SAR.pop()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec5(n))
        n = self.leaky(self.dec5b(n))

        n = self.upscale2d(n)
        if self.method != 'SAR':
          skips = skips_SAR.pop()+ skips_sup.pop()
        else:
          skips = skips_SAR.pop()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec4(n))
        n = self.leaky(self.dec4b(n))

        n = self.upscale2d(n)
        if self.method != 'SAR':
          skips = skips_SAR.pop()+ skips_sup.pop()
        else:
          skips = skips_SAR.pop()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec3(n))
        n = self.leaky(self.dec3b(n))

        n = self.upscale2d(n)
        if self.method != 'SAR':
          skips = skips_SAR.pop()+ skips_sup.pop()
        else:
          skips = skips_SAR.pop()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec2(n))
        n = self.leaky(self.dec2b(n))

        n = self.upscale2d(n)
        if self.method != 'SAR':
          skips = skips_SAR.pop()+ skips_sup.pop()
        else:
          skips = skips_SAR.pop()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec1a(n))
        n = self.leaky(self.dec1b(n))

        n = self.dec1(n)

        return x_SAR-n

    def loss_function(self,output,target,batch_size):
      """ Defines and runs the loss function

      Parameters
      ----------
      output :
      target :
      batch_size :

      Returns
      ----------
      loss: float
          The value of loss given your output, target and batch_size

      """
      # ----- loss -----
      M = 10.089038980848645
      m = -1.429329123112601
      # ----- loss -----
      log_hat_R = 2*(output*(M-m)+m)
      hat_R = torch.exp(log_hat_R)+1e-6 # must be nonzero
      b_square = torch.square(target)
      loss = (1/batch_size)*torch.mean( 0.5*log_hat_R+b_square/hat_R  ) #+ tf.losses.get_regularization_loss()
      return loss

    def training_step(self, batch,batch_number):

      """ Train the model with the training set

      Parameters
      ----------
      batch : a subset of the training date
      batch_number : ID identifying the batch

      Returns
      -------
      loss : float
        The value of loss given the batch

      """
      M = 10.089038980848645
      m = -1.429329123112601

      x, y, im = batch
      x=x.to(self.device)
      y=y.to(self.device)
      im=im.to(self.device)


      if (batch_number%2==0):
        x=(torch.log(torch.square(x)+1e-3)-2*m)/(2*(M-m))
        if self.method == 'SAR':
          out = self.forward(torch.unsqueeze(x,-1),self.batch_size)
        else:
          out = self.forward(torch.cat((torch.unsqueeze(x,-1),im),-1) ,self.batch_size)
        loss = self.loss_function(out, y,self.batch_size)

      else:
        y=(torch.log(torch.square(y)+1e-3)-2*m)/(2*(M-m))
        if self.method == 'SAR':
          out = self.forward(torch.unsqueeze(y,-1),self.batch_size)
        else:
          out = self.forward(torch.cat((torch.unsqueeze(y,-1),im),-1),self.batch_size)
        loss = self.loss_function(out,x,self.batch_size)

      return loss

    def validation_step(self, batch,image_num,epoch_num,eval_files,eval_set,sample_dir):
      """ Test the model with the validation set

      Parameters
      ----------
      batch : a subset of data
      image_num : an ID identifying the feeded image
      epoch_num : an ID identifying the epoch
      eval_files : .npy files used for evaluation in training
      eval_set : directory of dataset used for evaluation in training

      Returns
      ----------
      output_clean_image : a np.array

      """


      image_real_part,image_imaginary_part,im = batch

      image_real_part=image_real_part.to(self.device)
      image_imaginary_part=image_imaginary_part.to(self.device)
      im=im.to(self.device)
      # Normalization
      image_real_part_normalized=(torch.log(torch.square(image_real_part)+1e-3)-2*m)/(2*(M-m))
      image_imaginary_part_normalized=(torch.log(torch.square(image_imaginary_part)+1e-3)-2*m)/(2*(M-m))

      if self.method == 'SAR':
        out_real = self.forward(torch.unsqueeze(image_real_part_normalized,-1),self.eval_batch_size)
        out_imaginary = self.forward(torch.unsqueeze(image_imaginary_part_normalized,-1),self.eval_batch_size)
      else:
        out_real = self.forward( torch.cat((image_real_part_normalized,im),-1),self.eval_batch_size)
        out_imaginary = self.forward(torch.cat((image_imaginary_part_normalized,im),-1),self.eval_batch_size)

      output_clean_image = 0.5*(np.square(denormalize_sar(out_real.cpu().numpy()))+np.square(denormalize_sar(out_imaginary.cpu().numpy())))
      # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

      noisyimage = np.squeeze(np.sqrt(np.square(image_real_part.cpu().numpy())+np.square(image_imaginary_part.cpu().numpy())))
      outputimage = np.sqrt(np.squeeze(output_clean_image))

      print('Denoised image %d'%(image_num))

      # rename and save
      imagename = eval_files[image_num].replace(eval_set, "")
      imagename = imagename.replace('.npy', '_epoch_' + str(epoch_num) + '.npy')

      save_sar_images(outputimage, noisyimage, imagename,sample_dir)

      return output_clean_image

    def test_step(self, im, image_num, test_files, test_set, test_dir):

        pat_size = 256

        stride = 64

        # Pad the image
        # im on gpu ie tensor




        image_real_part,image_imaginary_part,opt = im
        im_h_start, im_w_start = image_real_part.size(dim=2), image_real_part.size(dim=3)
        image_real_part=image_real_part.to(self.device)
        image_imaginary_part=image_imaginary_part.to(self.device)
        opt=opt.to(self.device)

        # Normalization
        image_real_part_normalized=(torch.log(torch.square(image_real_part)+1e-3)-2*m)/(2*(M-m))
        image_imaginary_part_normalized=(torch.log(torch.square(image_imaginary_part)+1e-3)-2*m)/(2*(M-m))


        im_h, im_w = image_real_part.size(dim=2), image_real_part.size(dim=3)

        count_image = np.zeros((im_h, im_w))
        out = np.zeros((im_h, im_w))

        if im_h==pat_size:
                x_range = list(np.array([0]))
        else:
            x_range = list(range(0, im_h - pat_size, stride))
            if (x_range[-1] + pat_size) < im_h: x_range.extend(range(im_h - pat_size, im_h - pat_size + 1))

        if im_w==pat_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0, im_w - pat_size, stride))
            if (y_range[-1] + pat_size) < im_w: y_range.extend(range(im_w - pat_size, im_w - pat_size + 1))


        #testing by patch
        for x in x_range:
                for y in y_range:
                    patch_test_real = image_real_part_normalized[: , x:x + pat_size, y:y + pat_size, :]
                    patch_test_imag = image_imaginary_part_normalized[: , x:x + pat_size, y:y + pat_size, :]
                    patch_test_opt = opt[: , x:x + pat_size, y:y + pat_size, :]

                    tmp_real = self.forward(torch.cat((patch_test_real,patch_test_opt),dim=2), self.eval_batch_size)
                    tmp_imag = self.forward(torch.cat((patch_test_real,patch_test_opt),dim=2), self.eval_batch_size)

                    tmp = 0.5*(np.square(denormalize_sar(tmp_real.cpu().numpy()))+np.square(denormalize_sar(tmp_imag.cpu().numpy())))

                    out[x:x + pat_size, y:y + pat_size] = out[x:x + pat_size, y:y + pat_size] + tmp

                    count_image[x:x + pat_size, y:y + pat_size] = count_image[x:x + pat_size, y:y + pat_size] + np.ones((pat_size, pat_size))


        out = np.sqrt(out/count_image)
        #out is de-normalized



        imagename = test_files[image_num].replace(test_set, "")

        print('Denoised image %d'%(image_num))

        noisy = torch.sqrt(image_real_part**2+image_imaginary_part**2)
        save_sar_images(out, np.squeeze(np.asarray(noisy.cpu().numpy())), imagename, test_dir)
