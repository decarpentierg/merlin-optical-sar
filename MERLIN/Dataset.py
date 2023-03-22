import numpy as np
import torch
from glob import glob

from utils import symetrisation_patch


class Dataset(torch.utils.data.Dataset):
    "characterizes a dataset for pytorch"

    def __init__(self, patche, method):
        self.patches = patche
        self.method = method
        # self.patches_MR = patches_MR
        # self.patches_CT = patches_CT

    def __len__(self):
        "denotes the total number of samples"
        return len(self.patches)

    def __getitem__(self, index):
        "Generates one sample of data"
        # select sample
        batch_real = self.patches[index, :, :, 0]
        batch_imag = self.patches[index, :, :, 1]

        x = torch.tensor(batch_real)
        y = torch.tensor(batch_imag)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        if self.method != "SAR":
            batch_sup = self.patches[index, :, :, 2:]
            z = torch.tensor(batch_sup)

            # normalize z on each channel
            for i in range(z.shape[-1]):
                z[:, :, i] = (z[:, :, i] - torch.min(z[:, :, i])) / (
                    torch.max(z[:, :, i]) - torch.min(z[:, :, i])
                )
            z = z.unsqueeze(0)

        else:
            z = 0

        return x, y, z


class ValDataset(torch.utils.data.Dataset):
    "characterizes a dataset for pytorch"

    def __init__(self, test_set, method):
        self.files = glob(test_set + "/*.npy")
        self.method = method
        # self.patches_MR = patches_MR
        # self.patches_CT = patches_CT

    def __len__(self):
        "denotes the total number of samples"
        return len(self.files)

    def __getitem__(self, index):
        "Generates one sample of data"
        # select sample
        # eval_data = load_sar_images(self.files)
        if not isinstance(self.files, list):
            im = np.load(self.files)
            eval_data = np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), np.size(im, 2))
        else:
            eval_data = []
            for file in self.files:
                im = np.load(file)
                eval_data.append(
                    np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), np.size(im, 2))
                )
        current_test = eval_data[index]

        current_test[0, :, :, :] = symetrisation_patch(current_test[0, :, :, :])
        image_real_part = (current_test[:, :, :, 0]).reshape(
            current_test.shape[0], current_test.shape[1], current_test.shape[2], 1
        )
        image_imag_part = (current_test[:, :, :, 1]).reshape(
            current_test.shape[0], current_test.shape[1], current_test.shape[2], 1
        )
        if self.method != "SAR":
            image_sup = (current_test[:, :, :, 2:]).reshape(
                current_test.shape[0],
                current_test.shape[1],
                current_test.shape[2],
                current_test.shape[3] - 2,
            )
            # normalize image_sup on each channel
            for i in range(image_sup.shape[-1]):
                image_sup[:, :, :, i] = (
                    image_sup[:, :, :, i] - np.min(image_sup[:, :, :, i])
                ) / (np.max(image_sup[:, :, :, i]) - np.min(image_sup[:, :, :, i]))
        else:
            image_sup = 0

        return (
            torch.tensor(image_real_part).type(torch.float),
            torch.tensor(image_imag_part).type(torch.float),
            torch.tensor(image_sup).type(torch.float),
        )
