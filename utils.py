import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

## show a single image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    # print(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

## show a grid (batch) of images
def imgrid(loader):
    ## get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)
    ## show images
    imshow(torchvision.utils.make_grid(images))

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def visualize_model(device, model, loader, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 9))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 5, images_so_far)
                ax.axis('off')
                item = preds[j].item()
                ax.set_title('Healthy (0)' if item == 0 else 'Unhealthy (1)') #class_names[item]
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# Define the method to classify one image
def classify_image(model, image_path, class_labels):
    # Define preprocessing transform
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move image tensor to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Get predicted class
    _, predicted = torch.max(output, 1)
    predicted_class = class_labels[predicted.item()]

    return predicted_class