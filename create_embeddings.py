import os
import click
import torch
from torchvision import datasets, transforms
from networks.resnet_big import SupConResNet  # Assuming this is where your model is defined
# from util import AverageMeter  # Assuming this is defined in your util module
from util import generate_embeddings_filename

# Function to load the dataset
def load_dataset(dataset_name, num_embeddings_per_class):
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='./datasets/', train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root='./datasets/', train=True, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if num_embeddings_per_class < 0:
        return dataset
    
    assert num_embeddings_per_class > 0, num_embeddings_per_class

    # Create a class-balanced subset
    class_indices = {i: [] for i in range(10 if dataset_name == 'cifar10' else 100)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []
    for indices in class_indices.values():
        selected_indices.extend(indices[:num_embeddings_per_class])

    return torch.utils.data.Subset(dataset, selected_indices)

# Function to generate embeddings
def generate_embeddings(model, dataloader, use_head):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.cuda()
            if use_head:
                features = model(images)  # Use the full model output
            else:
                features = model.encoder(images)  # Use only the encoder output
            embeddings.append(features.cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(embeddings), torch.cat(labels)


@click.command()
@click.option('--model_type', type=click.Choice(['SimCLR', 'SupCon', 'TargetVector'], case_sensitive=False), required=True, help='Type of model to use.')
@click.option('--model_architecture', type=click.Choice(['resnet18', 'resnet34', 'resnet50'], case_sensitive=False), required=True, help='Model architecture to use.')
@click.option('--dataset', type=click.Choice(['cifar10', 'cifar100'], case_sensitive=False), required=True, help='Dataset to use.')
@click.option('--num_embeddings_per_class', type=int, default=-1, help='Number of embeddings to generate per class. Use None to use the entire dataset.')
@click.option('--ckpt', type=str, required=True, help='Path to the pre-trained model checkpoint.')
@click.option('--output_dir', type=str, default='./embeddings', help='Directory to save the generated embeddings.')
@click.option('--head', is_flag=True, help='Use the full model output (including head) instead of just the encoder output.')
@click.option('--norm', type=str, default='batchnorm2d', help='Normalization layer to use.')

def main(model_type, model_architecture, dataset, num_embeddings_per_class, ckpt, output_dir, head, norm):
    # Load the dataset
    dataset_name = dataset
    dataset = load_dataset(dataset_name, num_embeddings_per_class)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    # Load the model
    model = SupConResNet(name=model_architecture, norm=norm).cuda()  # Adjust if using a different model class
    loaded_checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = loaded_checkpoint['model']
    epoch = loaded_checkpoint['epoch']
    del loaded_checkpoint
    model.load_state_dict(state_dict)
    
    # Generate embeddings and labels
    embeddings, labels = generate_embeddings(model, dataloader, head)

    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)
    filename = generate_embeddings_filename(model_type, dataset_name, model_architecture, num_embeddings_per_class, head, epoch)

    # Save as a dictionary
    torch.save({'embeddings': embeddings, 'labels': labels, 'epoch': epoch}, os.path.join(output_dir, filename))
    
    print(f"Embeddings saved to {output_dir}/{filename}")

if __name__ == '__main__':
    main()
