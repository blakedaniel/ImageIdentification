import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

class ImageClassifactionTuner:
    def __init__(self, dir:str='./classification_data') -> None:
        self.data_dir = dir
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    def define_image_transformation(self, crop:int=224, resize:int=256):
        """
        Define the image transformations for the dataset.

        Args:
            crop (int): The size of the cropped image. Default is 224.
            resize (int): The size of the resized image. Default is 256.

        Returns:
            None
        """
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
    def load_data(self):
            """
            Loads the image datasets and creates dataloaders for training and validation.
            
            Returns:
                None
            """
            self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in ['train', 'val']}
            self.dataloaders = {x: DataLoader(self.image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}
            self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
            self.class_names = self.image_datasets['train'].classes
        
    def load_model(self, model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT), output_size:int=2,):
        """
        Loads a pre-trained ResNet18 model and modifies the fully connected layer for the specified output size.
        
        Args:
            model (torchvision.models.ResNet): The pre-trained ResNet18 model to load. Defaults to models.resnet18(weights=models.ResNet18_Weights.DEFAULT).
            output_size (int): The number of output classes. Defaults to 2.
        """
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_size)
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    def train_model(self, num_epochs:int=10):
            """
            Trains the model for a specified number of epochs.

            Args:
                num_epochs (int): The number of epochs to train the model (default: 10).
            """
            
            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, labels)
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                print()
        
    def save_model(self):
        """
        Saves the state dictionary of the model to a file named 'model.pth'.
        """
        torch.save(self.model.state_dict(), 'model.pth')
        
    def predict(self, image_path:str):
        """
        Predicts the class of an image using the trained model.

        Parameters:
        image_path (str): The path to the image file.

        Returns:
        str: The predicted class name.
        """
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()
        image = Image.open(image_path)
        image_tensor = self.data_transforms['val'](image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = image_tensor.to(self.device)
        output = self.model(input)
        index = output.data.cpu().numpy().argmax()
        return self.class_names[index]

if __name__ == '__main__':
    ict = ImageClassifactionTuner()
    ict.define_image_transformation()
    ict.load_data()
    ict.load_model()
    ict.train_model()
    ict.save_model()
    print(ict.predict('../tests/test_data/ALSTONIA.jpg'))
    print(ict.predict('../tests/test_data/Bay-Leaves-Vs.-Basil.jpeg'))
    
    