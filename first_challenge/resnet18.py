import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from sklearn import metrics
from collections import Counter

from utils import get_device, visualize_model, classify_image
# from datasets import AutoencoderCustomDataset
# from models import ConvAutoencoder
from earlystopping import EarlyStopping

print(torch.__version__)

seed_value= 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

t_loss = []
t_acc = []
v_loss = []
v_acc = []

def main():
    class_names={0: 'minus', 1: 'plus'}
    criterion = nn.CrossEntropyLoss()

    BATCH_SIZE=32

    ##################################
    ## TODO: move to datasets

    transformations = transforms.Compose([
        # transforms.Resize(512),
        # transforms.CenterCrop(512),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(45),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(root='/Users/verenaojeda/RICSE/notebooks/AnnotatedPatches_3/train_80_persent/', transform=transformations)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    test_dataset = torchvision.datasets.ImageFolder(root='/Users/verenaojeda/RICSE/notebooks/AnnotatedPatches_3/test_20_persent/', transform=transformations)

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)
    class_names={0: 'minus', 1: 'plus'}

    indices = np.arange(len(dataset.targets))
    np.random.shuffle(indices)
    total = len(indices)
    a = int(np.round(0.7 * total))

    train_idx = indices[0:a]
    val_idx = indices[a:total]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    test_indices = np.arange(len(test_dataset.targets))
    np.random.shuffle(test_indices)
    test_total = len(test_indices)
    test_idx = indices[0:test_total]
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=2)
    valloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=2)

    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
        num_workers=2)

    print('Total:', dict(Counter(dataset.targets)))

    train_classes = [dataset.targets[i] for i in train_idx]
    print('Train Set:', dict(Counter(train_classes)))

    val_classes = [dataset.targets[i] for i in val_idx]
    print('Validation Set:', dict(Counter(val_classes)))

    test_classes = [dataset.targets[i] for i in test_idx]
    print('Test Set:', dict(Counter(test_classes)))

    TRAIN_COUNT = len(train_idx)
    VAL_COUNT = len(val_idx)
    TEST_COUNT = len(test_idx)

    # total_dict = dict(Counter(dataset.targets))
    # MINUS=total_dict.get(0)
    # PLUS=total_dict.get(1)

    ##################################

    # Change number of epochs
    NUM_EPOCHS=2
    LEARNING_RATE=1e-2
    MOMENTUM=0.9
    
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names));

    device = get_device()
    model_ft = model_ft.to(device)

    # Try different optimizers
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE)
    # optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
    # optimizer_ft = torch.optim.RMSprop(model_ft.parameters(), lr=LEARNING_RATE)

    model_ft = train_model(device, model_ft, criterion, optimizer_ft, trainloader, valloader, testloader, TRAIN_COUNT, VAL_COUNT, TEST_COUNT, scheduler=None, num_epochs=NUM_EPOCHS)

    # visualize_model(device, model_ft, testloader, num_images=10)

    # # Example classification
    # image_path = '/content/gdrive/MyDrive/RICSE_DeepL/AnnotatedPatches_3/test_20_persent/minus/B22-240_0.00814.png'
    # class_labels = ['healthy', 'unhealthy']  # List of class labels corresponding to model output

    # predicted_class = classify_image(model_ft, image_path, class_labels)
    # print("Predicted class:", predicted_class)

    # Plot the training and validation loss
    plt.plot(t_loss, label='Training Loss')
    plt.plot(v_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    xt_acc = [acc.cpu().data for acc in t_acc]
    xv_acc = [acc.cpu().data for acc in v_acc]

    print()

    # Plot the training and validation acc
    plt.plot(xt_acc, label='Training Acc')
    plt.plot(xv_acc, label='Validation Acc')
    plt.title('Training and Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

def train_model(device, model, criterion, optimizer, trainloader, valloader, testloader, TRAIN_COUNT, VAL_COUNT, TEST_COUNT, scheduler=None, num_epochs=25):
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)

    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for images, labels in trainloader:
            # inputs = sample['image'].to(device, dtype=torch.float)
            # labels = sample['label'].to(device)
            inputs = images.to(device)
            labelss = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labelss)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labelss.data)

        if scheduler:
          scheduler.step()

        epoch_loss = running_loss / TRAIN_COUNT
        epoch_acc = running_corrects.double() / TRAIN_COUNT
        t_loss.append(epoch_loss)
        t_acc.append(epoch_acc)

        print('TRAIN - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
          best_acc = epoch_acc

        print()

        # VALIDATE MODEL
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
          for i, (images, labels) in enumerate(valloader, 0):
              # inputs = sample['image'].to(device, dtype=torch.float)
              # labels = sample['label'].to(device)
              inputs = images.to(device)
              labelss = labels.to(device)

              outputs = model(inputs)
              _, preds = torch.max(outputs, 1)
              val_loss = criterion(outputs, labelss)

              # statistics
              val_running_loss += val_loss.item() * inputs.size(0)
              val_running_corrects += torch.sum(preds == labelss.data)

        val_epoch_loss = val_running_loss / VAL_COUNT
        val_epoch_acc = val_running_corrects.double() / VAL_COUNT
        v_loss.append(val_epoch_loss)
        v_acc.append(val_epoch_acc)

        print('VALIDATION - Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_epoch_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:4f}'.format(best_acc))

    # TEST MODEL
    model.eval()
    test_acc = 0.0
    correct_count = 0

    predlabels=torch.zeros(0,dtype=torch.long, device='cpu')
    truelabels=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
      for i, (images, labels) in enumerate(testloader, 0):
          # inputs = sample['image'].to(device, dtype=torch.float)
          # labels = sample['label'].to(device)
          inputs = images.to(device)
          labelss = labels.to(device)
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          corrects = (torch.max(outputs, 1)[1].view(labelss.size()).data == labelss.data).sum()
          correct_count += corrects.item()

          predlabels=torch.cat([predlabels,preds.view(-1).cpu()])
          truelabels=torch.cat([truelabels,labelss.view(-1).cpu()])

      test_acc = correct_count / TEST_COUNT

    print()
    print('Test Accuracy: {:4f}'.format(test_acc))

    # Confusion matrix
    conf_mat=metrics.confusion_matrix(truelabels.numpy(), predlabels.numpy())
    # print('Confusion Matrix')
    # print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    # print('Class Accuacy')
    # print(class_accuracy)

    # Sensitivity & specificity
    tn, fp, fn, tp = conf_mat.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr, tpr, thresholds = metrics.roc_curve(truelabels.numpy(), predlabels.numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('Sensitivity: {:4f}'.format(sensitivity))
    print('Specificity: {:4f}'.format(specificity))
    print('AUC: {:4f}'.format(auc))

    # Plot ROC
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return model
        
if __name__ == '__main__':
    main()