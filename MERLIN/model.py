import numpy as np
import torch

from utils import denormalize_sar, save_sar_images


# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
M = 10.089038980848645
m = -1.429329123112601


class AE(torch.nn.Module):
    """Auto-Encoder/U-Net
    
    Attributes
    ----------
    batch_size: int
        batch size
    
    eval_batch_size: int
        batch size for evaluation
    
    device: str
        name of device to use
    
    method: str
        either 'SAR' or 'SAR+OPT' or 'SAR+SAR' or 'SAR+OPT+SAR'
        defines what images are given as input to the neural network
    """

    def __init__(self, batch_size, eval_batch_size, device, method):
        """Class constructor
        
        Parameters
        ----------
        See class docstring.
        """
        super().__init__()

        # Set attributes
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.method = method

        # Compute number of auxiliary channels to add, depending on the chosen method
        self.n_channels_sup = {'SAR':0, 'SAR+OPT': 1, 'SAR+SAR': 1, 'SAR+OPT+SAR': 2}[method]
        
        # ------------
        # Build layers
        # ------------

        # Max-pooling
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Activation function
        self.leaky = torch.nn.LeakyReLU(0.1)

        conv_kwargs = {'kernel_size': (3, 3), 'stride': (1, 1), 'padding': 'same'}

        # Main encoder
        self.enc0 = torch.nn.Conv2d(in_channels=1, out_channels=48, **conv_kwargs)
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
        
        # Auxiliary encoder. Processes additional images (optical, other SAR images, ...)
        if method != 'SAR':
            self.enc0_opt = torch.nn.Conv2d(in_channels=self.n_channels_sup, out_channels=48, **conv_kwargs)
            self.enc1_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
            self.enc2_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
            self.enc3_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
            self.enc4_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
            self.enc5_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
            self.enc6_opt = torch.nn.Conv2d(in_channels=48, out_channels=48, **conv_kwargs)
        
        # Decoder
        self.dec5 = torch.nn.Conv2d(in_channels=96, out_channels=96, **conv_kwargs)
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, **conv_kwargs)
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, **conv_kwargs)
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, **conv_kwargs)
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, **conv_kwargs)
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, **conv_kwargs)
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, **conv_kwargs)
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, **conv_kwargs)
        if self.method != 'SAR+OPT+SAR':
            self.dec1a = torch.nn.Conv2d(in_channels=97, out_channels=64, **conv_kwargs)
        else:
            self.dec1a = torch.nn.Conv2d(in_channels=98, out_channels=64, **conv_kwargs)
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, **conv_kwargs)
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=1, **conv_kwargs)

        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self, x, batch_size):
        """Forward propagation for an image x

        Parameters
        ----------
        x : np.array
        a numpy array containing an image

        Returns
        ----------
        x - n : np.array
        a numpy array containing the denoised image i.e the image itself minus the noise
        """
        # Reshape input
        x_SAR = torch.reshape(x[:, :, :, :, 0], [batch_size, 1, 256, 256])

        # ------------
        # Main encoder
        # ------------

        # List for skip connections
        skips_SAR = [x_SAR]

        n_SAR = self.leaky(self.enc0(x_SAR))
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

        # -----------------
        # Auxiliary encoder
        # -----------------

        if self.method != 'SAR':
            # List for skip 
            x_sup = torch.reshape(x[:, :, :,:, 1:], [batch_size, self.n_channels_sup, 256, 256])
            skips_sup = [x_sup]

            n_sup = self.leaky(self.enc0_opt(x_sup))
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

        # ----------------------------------
        # Combine outputs from both encoders
        # ----------------------------------
        if self.method == 'SAR':
            n = n_SAR
        else:
            n = n_SAR + n_sup

        # -------
        # Decoder
        # -------
        def get_skips():
            """Get current skip connections."""
            if self.method != 'SAR':
                skips = skips_SAR.pop() + skips_sup.pop()
            else:
                skips = skips_SAR.pop()
            return skips
        

        n = self.upscale2d(n)
        skips = get_skips()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec5(n))
        n = self.leaky(self.dec5b(n))

        n = self.upscale2d(n)
        skips = get_skips()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec4(n))
        n = self.leaky(self.dec4b(n))

        n = self.upscale2d(n)
        skips = get_skips()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec3(n))
        n = self.leaky(self.dec3b(n))

        n = self.upscale2d(n)
        skips = get_skips()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec2(n))
        n = self.leaky(self.dec2b(n))

        n = self.upscale2d(n)
        skips = get_skips()
        n = torch.cat((n, skips), dim=1)
        n = self.leaky(self.dec1a(n))
        n = self.leaky(self.dec1b(n))

        n = self.dec1(n)

        return x_SAR - n

    def loss_function(self, output, target, batch_size):
        """Computes the loss function.

        Parameters
        ----------
        output :
        target :
        batch_size :

        Returns
        ----------
        loss: float
        """
        log_hat_R = 2 * (output * (M - m) + m)
        hat_R = torch.exp(log_hat_R) + 1e-6  # must be nonzero
        b_square = torch.square(target)
        loss = (1 / batch_size) * torch.mean(0.5 * log_hat_R + b_square / hat_R)
        return loss

    def training_step(self, batch, batch_number):
        """Train the model with the training set

        Parameters
        ----------
        batch : a subset of the training date
        batch_number : ID identifying the batch

        Returns
        -------
        loss : float
          The value of loss given the batch
        """
        x, y, im = batch
        x = x.to(self.device)
        y = y.to(self.device)
        im = im.to(self.device)

        # Depending on the batch number, we predict either the imaginary part from the real part or
        # the real part from the imaginary part.
        if batch_number % 2 == 0:
            known, to_predict = x, y
        else:
            known, to_predict = y, x

        input = (torch.log(torch.square(known) + 1e-3) - 2 * m) / (2 * (M - m))
        input = torch.unsqueeze(input, -1)

        if self.method != 'SAR':
            input = torch.cat((input, im), -1)

        out = self.forward(input, self.batch_size)
        loss = self.loss_function(out, to_predict, self.batch_size)

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
