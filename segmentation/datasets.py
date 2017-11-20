from torch.utils.data import Dataset
from skimage.io import imread
from os.path import join
from glob import glob
from PIL import Image

class BroadDataset(Dataset):
    """Broad Dataset BBBC020."""

    def __init__(self, root_dir, mode, joint_transform=None, input_transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and ground truth.
            mode (string): One of ['train', 'test', 'val'].
            joint_transform (callable, optional): Optional transform to be applied
                to both input and mask.
            input_transform (callable, optional): Optional transform to be applied
                on input.
            target_transform (callable, optional): Optional transform to be applied
                on mask.
        """
        self.root_dir = root_dir
        assert(mode in ['train', 'test', 'val'])
        self.mode = mode
        self.names = glob(join(self.root_dir, self.mode, 'image_*.tif'))
        self.masks = glob(join(self.root_dir, self.mode, 'mask_*.tif'))
        assert(len(self.names) == len(self.masks))
        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        #name = #
        image = imread(self.names[idx])
        image[int(image.shape[0]/2)-10:int(image.shape[0]/2)+10,int(image.shape[1]/2)-10:int(image.shape[1]/2)+10,:] = [255, 255, 255]
        image = Image.fromarray(image)
        mask = Image.fromarray(imread(self.masks[idx]))

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask