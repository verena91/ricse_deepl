import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import get_device
from datasets import AutoencoderCustomDataset
from models import ConvAutoencoder

print(torch.__version__)

seed_value= 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
              
def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    metadata_path = '/Users/verenaojeda/RICSE/cluster_data/fhome/mapsiv/QuironHelico/CroppedPatches/metadata_test.csv'
    images_dir = '/Users/verenaojeda/RICSE/cluster_data/fhome/mapsiv/QuironHelico/CroppedPatches/'
    
    # Create an instance of CustomDataset
    dataset = AutoencoderCustomDataset(csv_file=metadata_path, root_dir=images_dir, transform=transform)

    # # Check data
    # image, label = dataset[0]
    # print("Image shape:", image.shape)
    # print("Label:", label)

    BATCH_SIZE=64
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    model = ConvAutoencoder()
    device = get_device()
    model.to(device)

    #Loss function
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #Epochs
    n_epochs = 10
    train_loss_arr = []

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0

        #Training
        # for data in train_loader:
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(batch_idx)
            # image = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)

        train_loss = train_loss/len(train_loader)
        train_loss_arr.append(train_loss)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


    ## Save Entire Model
    torch.save(model, f'/Users/verenaojeda/RICSE/cluster_data/saved_models/autoencoder{datetime.time()}')

    # ## Load Entire Model
    # model = torch.load('/content/gdrive/MyDrive/RICSE_DeepL/saved_models/autoencoderX')
    # model.eval()

    # Plot the training and validation loss
    plt.plot(train_loss_arr, label='Training Loss')
    # plt.plot(v_loss, label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
if __name__ == '__main__':
    main()