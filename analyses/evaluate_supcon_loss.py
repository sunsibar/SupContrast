import os
import click
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from networks.resnet_big import SupConResNet
from losses import SupConLoss, TargetVectorMSELoss
from util import TwoCropTransform, AverageMeter

@click.command()
@click.option('--model_type', type=click.Choice(['SimCLR', 'SupCon', 'TargetVector'], case_sensitive=False), required=True)
@click.option('--model_architecture', type=click.Choice(['resnet18', 'resnet34', 'resnet50'], case_sensitive=False), required=True)
@click.option('--dataset', type=click.Choice(['cifar10', 'cifar100'], case_sensitive=False), required=True)
@click.option('--batch_size', type=int, default=2048)
@click.option('--num_workers', type=int, default=8)
@click.option('--temperature', type=float, default=0.07)
@click.option('--ckpt', type=str, required=True, help='Path to model checkpoint')
@click.option('--data_folder', type=str, default='./datasets')
@click.option('--head', is_flag=True, help='Use the head of the model')

def main(model_type, model_architecture, dataset, batch_size, num_workers, temperature, ckpt, data_folder, head):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up data transforms
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:  # cifar100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    # Define transforms for train and test
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Create datasets
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_folder, train=True,
                                       transform=TwoCropTransform(train_transform),
                                       download=True)
        test_dataset = datasets.CIFAR10(root=data_folder, train=False,
                                      transform=TwoCropTransform(test_transform),
                                      download=True)
    else:
        train_dataset = datasets.CIFAR100(root=data_folder, train=True,
                                        transform=TwoCropTransform(train_transform),
                                        download=True)
        test_dataset = datasets.CIFAR100(root=data_folder, train=False,
                                       transform=TwoCropTransform(test_transform),
                                       download=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # Create model and load checkpoint
    model = SupConResNet(name=model_architecture)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True

    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Create criterion
    criterion = SupConLoss(temperature=temperature)
    
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    def evaluate(loader, split='train'):
        losses = AverageMeter()
        
        with torch.no_grad():
            for idx, (images, labels) in enumerate(loader):
                images = torch.cat([images[0], images[1]], dim=0)
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]

                # Forward pass
                features = model(images)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                
                # Compute loss
                loss = criterion(features, labels)
                
                losses.update(loss.item(), bsz)
                
                if (idx + 1) % 100 == 0:
                    print(f'{split}: [{idx + 1}/{len(loader)}]\t'
                          f'Loss {losses.val:.3f} ({losses.avg:.3f})')

        return losses.avg

    # Evaluate on train and test sets
    print("Evaluating on training set...")
    train_loss = evaluate(train_loader, 'train')
    print("Evaluating on test set...")
    test_loss = evaluate(test_loader, 'test')

    print(f"\nFinal Results:")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Criterion: {criterion}, has type {type(criterion)}")

if __name__ == '__main__':
    main()