import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize((64, 64)),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize((64, 64)),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class OCTDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.cond_to_label = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpeg')]

        # self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.labels = [self.cond_to_label[filename.split('-')[0].split('/')[-1]] for filename in self.filenames]
        self.cnv_filenames = [f for i, f in enumerate(self.filenames) if self.labels[i] == 0]
        self.dme_filenames = [f for i, f in enumerate(self.filenames) if self.labels[i] == 1]
        self.drusen_filenames = [f for i, f in enumerate(self.filenames) if self.labels[i] == 2]
        self.normal_filenames = [f for i, f in enumerate(self.filenames) if self.labels[i] == 3]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        # return len(self.filenames)
        return len(self.cnv_filenames) * 4

    def __getitem__(self, idx):
        # fix to balance dataset 
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image_name = None
        label = None
        if idx % 4 == 0: 
            image_name = self.cnv_filenames[idx % len(self.cnv_filenames)]
            label = 0
        elif idx % 4 == 1:
            image_name = self.dme_filenames[idx % len(self.dme_filenames)]
            label = 1
        elif idx % 4 == 2: 
            image_name = self.drusen_filenames[idx % len(self.dme_filenames)]
            label = 2
        else: 
            image_name = self.normal_filenames[idx % len(self.normal_filenames)]
            label = 3
        image = Image.open(image_name).convert("RGB")  # PIL image
        # image = Image.open(self.filenames[idx]).convert("RGB")  # PIL image
        image = self.transform(image)
        # return image, self.labels[idx]
        return image, label


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            if split == 'train':
                dl = DataLoader(OCTDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(OCTDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
