import argparse
from dataset import create_dataloader
from torch import cuda
from tqdm.auto import tqdm
import torch
from model import create_model
from utils import *
import numpy as np
from torch.autograd import Variable


def parse_option():
    parser = argparse.ArgumentParser('Tiles Mini Dataset', add_help=False)
    # Defining Device
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    # Train Options
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--model_type', type=str, default='resnet')
    # Dataset Options
    parser.add_argument('--split_ratios', type=list, default=[0.8, 0.1, 0.1])
    parser.add_argument("--mnist", type=bool, default=False, choices=[True, False])
    # Directory Options
    parser.add_argument('--data_path', type=str, help='path to dataset', default='data/tiles_mini')
    parser.add_argument('--output_dir', default='outputs', type=str, metavar='PATH')
    parser.add_argument('--check_dir', default='checkpoints', type=str, metavar='PATH')
    # Data Augmentation Options
    parser.add_argument('--mix_up', default=False, type=bool)
    parser.add_argument('--rand_augment', default=False, type=bool)

    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--resume', help='resume from checkpoint', default=False, choices=[True, False])

    parser.add_argument("--model_name", type=str, default="ResNet18_with_TL_and_no_AUG")

    args = parser.parse_args()
    
    print('Args:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()

    return args


def mixup_data(x, y, alpha=0.4, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device == 'cuda':
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# training
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        if args.mix_up:
            image, targets_a, targets_b, lam = mixup_data(image, labels, device=DEVICE)
            image, targets_a, targets_b = map(Variable, (image, targets_a, targets_b))
            outputs = model(image)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(image)
            loss = criterion(outputs, labels)
        
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        loss.backward()

        optimizer.step()
    
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        counter += 1
        
        image, labels = data
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image)

        loss = criterion(outputs, labels)
        valid_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        valid_running_correct += (preds == labels).sum().item()
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))
    return epoch_loss, epoch_acc


def inference(model, test_loader, args):
    check = torch.load(f'{args.output_dir}/best_{args.model_name}.pth', args.device)
    model.load_state_dict(check['model_state_dict'])
    print('\nBest Weights Loaded')

    model.eval()
    print('Inference')
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        counter += 1
        
        image, labels = data
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image)

        loss = criterion(outputs, labels)
        test_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        test_running_correct += (preds == labels).sum().item()
        
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(test_loader.dataset))

    print(f"Test loss: {epoch_loss:.3f}, Test acc: {epoch_acc:.3f}")

    return epoch_loss, epoch_acc



if __name__ == '__main__':
    args = parse_option()
    EPOCHS = args.epochs
    DEVICE = args.device
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir

    train_loader, valid_loader, test_loader, num_classes = create_dataloader(args)
    args.num_classes = num_classes

    print('\nNumber of train samples in train dataset:', len(train_loader.dataset))
    print('Number of train samples in Validation dataset:', len(valid_loader.dataset))
    print('Number of train samples in Test dataset:', len(test_loader.dataset))
    print()

    model = create_model(args).to(DEVICE)

    n = 0
    for p in model.parameters():
        p.requires_grad_(True)
        n += p.numel()
    print('Number of parameters:', n)
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    if args.resume:
        model, optimizer, curr_epoch, train_loss, valid_loss, train_accuracy, valid_accuracy = load_checkpoint(model, optimizer, args)
        save_best_model = SaveBestModel(min(valid_loss))
    else:
        train_loss, valid_loss = [], []
        train_accuracy, valid_accuracy = [], []
        save_best_model = SaveBestModel()
        curr_epoch = 0

    for epoch in range(curr_epoch, EPOCHS):
        print(f"Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_accuracy.append(train_epoch_acc)
        valid_accuracy.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

        save_best_model(valid_epoch_loss, model, optimizer, epoch, OUTPUT_DIR, 
                        MODEL_NAME, train_loss, valid_loss, train_accuracy, valid_accuracy)
        save_model(model, optimizer, epoch, OUTPUT_DIR, MODEL_NAME, 
                   train_loss, valid_loss, train_accuracy, valid_accuracy)
        save_plots(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch, OUTPUT_DIR, MODEL_NAME)

        print('-'*50)

    inference(model, test_loader, args)
