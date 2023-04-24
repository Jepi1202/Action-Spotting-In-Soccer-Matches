import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import sqlite3

class FootballDataset(Dataset):
    """
    Class to prepare the data for a neural network in pyTorch. The database is composed of 3 tables:
    - Video: to save the video information
    - Sequence: to save the sequence information labelled with the soccer actions and record the sound of the sequence
    - Image: to save the image information of the sequence
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_images(db_path: str, classes: list = ["Corner"], type_: str = "Training") -> list:
        """
        Connects to the database and returns the images and labels.

        Args:
            db_path (str): Path to the database.
            classes (list, optional): List of classes to be considered. Defaults to ["Corner"].
            type_ (str, optional): Type of the video. Defaults to "Training".
        Returns:
            list: List of tuples with the image path and the label.
        """
        # connect to the database
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # get the images and labels
        c.execute("SELECT im.Path, seq.Label FROM IMAGE im INNER JOIN Sequence seq ON im.SequencePath = seq.Path_sequence INNER JOIN VIDEO vid ON seq.VideoPath = vid.Path_video WHERE vid.training_stage = ?", (type_,))

        data = c.fetchall()
        
        pred = []
        for val in data:
            if val[1] in classes:
                pred.append((val[0], val[1]))
            else:
                pred.append((val[0], "NoClass"))
        return pred


if __name__=="__main__":
    import torch
    import torchvision
    from torchvision import transforms

    # Define data transformations for data augmentation and normalization
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define the dataset and dataloader
    data_list = FootballDataset.get_images(db_path="Mettre path de la DB", type_="Training") # type_ peut être "Training" ou "Test" ou "Validation"
    dataset = FootballDataset(data_list=data_list, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Define the CNN model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 56 * 56, 10),
        torch.nn.Softmax(dim=1)
    )

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training loss after every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


