import torch
from torchvision import datasets, transforms
import os

def process(output_dir='./datasets/raw'):
    """
    Downloads and processes MNIST, merging train and test sets.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    def get_all_tensors(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        return next(iter(loader))

    x_train, y_train = get_all_tensors(train_set)
    x_test, y_test = get_all_tensors(test_set)
    
    all_x = torch.cat([x_train, x_test], dim=0)
    all_y = torch.cat([y_train, y_test], dim=0)
    
    processed_data = {
        'x': all_x,
        'y': all_y,
        'num_classes': 10
    }
    
    output_path = os.path.join(output_dir, 'mnist_raw.pt')
    torch.save(processed_data, output_path)
    return output_path
