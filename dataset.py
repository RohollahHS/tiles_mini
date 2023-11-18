'https://drive.google.com/file/d/1-89C2dX4pZEazjGa3JmIJ42RpoVReY4C/view?usp=sharing'

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data


def create_dataloader(args):
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = datasets.ImageFolder(root=f'{args.data_path}')

    train_size = int(args.split_ratios[0] * len(dataset))
    valid_size = int(args.split_ratios[1] * len(dataset))
    test_size = len(dataset) - (train_size + valid_size)

    train, valid, test = data.random_split(dataset, [train_size, valid_size, test_size])

    train_dataset = CustomDataset(train, train_transform)
    valid_dataset = CustomDataset(valid, valid_transform)
    test_dataset  = CustomDataset(test, valid_transform)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader  = data.DataLoader(test_dataset,  batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


class CustomDataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x = self.transform(self.dataset[index][0])
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)



# the training transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])