import torch
from resnet import resnet18


def create_model(args, num_classes):
    model = resnet18()

    try:
        check = torch.load(f'checkpoints/resnet50-11ad3fa6.pth')
        model.load_state_dict(check)
        print('\nUsing IMAGENET Weights as Initialization!\n')
    except:
        pass

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(args.device)

    for p in model.parameters():
        p.requires_grad_(True)

    n = 0
    for p in model.parameters():
        n += p.numel()
    
    print('Number of parameters:', n)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    return model, optimizer, criterion