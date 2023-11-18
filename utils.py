from torch import load, save
from numpy import zeros
import matplotlib.pyplot as plt
import pylab
import os
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_checkpoint(model, optimizer, args):
    
    checkpoint = load(f'{args.output_dir}/last_{args.model_name}.pth', args.device)

    model.load_state_dict(checkpoint['model_state_dict']).to(args.device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']).to(args.device)
    curr_epoch = checkpoint['epoch']

    train_loss = zeros(args.epochs)
    valid_loss = zeros(args.epochs)
    train_accuracy = zeros(args.epochs)
    valid_accuracy = zeros(args.epochs)

    train_loss[:curr_epoch]   = checkpoint['train_loss'][:curr_epoch]
    valid_loss[:curr_epoch]   = checkpoint['valid_loss'][:curr_epoch]
    train_accuracy[:curr_epoch] = checkpoint['train_accuracy'][:curr_epoch]
    valid_accuracy[:curr_epoch] = checkpoint['valid_accuracy'][:curr_epoch]

    print('\nChekcpoint Loaded Successfully!\n')
    
    return model, optimizer, curr_epoch, train_loss, valid_loss, train_accuracy, valid_accuracy


def save_model(
    model,
    optimizer,
    epoch,
    save_dir,
    model_name,
    train_loss,
    valid_loss,
    train_accuray,
    valid_accuray,
):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_accuray": train_accuray,
            "valid_accuray": valid_accuray,
        },
        f"{save_dir}/last_{model_name}.pth",
    )


class SaveBestModel:
    def __init__(self) -> None:
        self.loss = float('inf')
    def __call__(
        self,
        loss,
        model,
        optimizer,
        epoch,
        save_dir,
        model_name,
        train_loss,
        valid_loss,
        train_accuray,
        valid_accuray,
        ):
            if loss < self.loss:
                self.loss = loss
                print(f"Saving best model for epoch: {epoch+1}\n")
                save(
                    {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_accuray": train_accuray,
                "valid_accuray": valid_accuray,
                    },
                f"{save_dir}/best_{model_name}.pth",
                )


def load_best_model(model, args):
     check = load(f'{args.output_dir}/best_{args.model_name}.pth')
     model.load_state_dict(check['model_state_dict'])
     print('\nBest Model Loaded.\n')
     return model


def save_loss_records(dir_path, file_name, loss=None, epoch=None, model_name=None):
    text_file = open(f'{dir_path}/{file_name}.txt', 'a')

    if (model_name == None) and (loss != None):
        text_file.write(f'{epoch}-{loss:.4f} | ')
    elif (model_name != None) and (loss == None):
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        text_file.write(f'\n{model_name} - {now}\n')
    
    text_file.close()


def save_plots(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch, save_dir, model_name):

    plt.figure()
    # pylab.xlim(0, epoch + 1)
    plt.plot(range(1, epoch + 1), train_loss[:epoch+1], label='train_loss')
    plt.plot(range(1, epoch + 1), valid_loss[:epoch+1], label='valid_loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.pdf'))
    plt.close()

    plt.figure()
    # pylab.xlim(0, epoch + 1)
    # pylab.ylim(0, 1)
    plt.plot(range(1, epoch + 1), train_accuracy[:epoch+1], label='train_accuracy')
    plt.plot(range(1, epoch + 1), valid_accuracy[:epoch+1], label='valid_accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{model_name}_accuracy.pdf'))
    plt.close()
