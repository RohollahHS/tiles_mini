import argparse
from dataset import create_dataloader
from torch import cuda
from tqdm.auto import tqdm
import torch
from model import create_model
from utils import *

def parse_option():
    parser = argparse.ArgumentParser('Tiles Mini Dataset', add_help=False)
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--split_ratios', type=list, default=[0.8, 0.1, 0.1])

    parser.add_argument('--file_name', type=str, help='tiles_mini', default='tickets')
    parser.add_argument('--data_path', type=str, help='path to dataset', default='data')
    parser.add_argument('--output_dir', default='outputs', type=str, metavar='PATH')

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--num_classes', default=6, type=int)

    parser.add_argument('--all_data', type=str, default=False, choices=[True, False])

    parser.add_argument('--resume', help='resume from checkpoint', default=False, choices=[True, False])

    parser.add_argument("--model_name", type=str, default="Debugging")

    args = parser.parse_args()
    
    print('Args:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()

    return args


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

        outputs = model(image)

        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        loss.backward()

        optimizer.step()
    
    epoch_loss = train_running_loss / counter
    epoch_acc = train_running_correct / len(trainloader.dataset)
    return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            counter += 1
            
            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(image)

            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = valid_running_correct / len(valid_loader.dataset)
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    args = parse_option()
    EPOCHS = args.epochs
    DEVICE = args.device
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir

    train_loader, valid_loader, test_loader = create_dataloader(args)

    model, optimizer, criterion = create_model(args, num_classes=6)

    if args.resume:
        model, optimizer, curr_epoch, train_loss, valid_loss, train_accuracy, valid_accuracy = load_checkpoint(model, optimizer, args)
        save_best_model = SaveBestModel(min(valid_loss))
    else:
        train_loss, valid_loss = [], []
        train_accuracy, valid_accuracy = [], []
        save_best_model = SaveBestModel()
        curr_epoch = 0

    for epoch in range(curr_epoch, EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
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

