import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_parquet(file_path):
    """Load a Parquet file and return its data."""
    df = pd.read_parquet(file_path)
    return df

def show_window(window_data):
    """Load a Window data and display it."""
    plt.figure(figsize=(12, 6))

    time = np.arange(window_data.shape[1]) / 128  # assuming 128 Hz sampling rate
    for i in range(window_data.shape[0]):  # iterate over channels
        plt.plot(time, window_data[i], label=f'Channel {i+1}')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Data for the First Window')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.show()

# Define LSTM model
class EpilepsyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EpilepsyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Use only the last output of the sequence
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def main():
    # TODO: merge with more metadata_x files
    annotation_file = '/Users/verenaojeda/RICSE/notebooks/sample_data/chb01_seizure_metadata_1.parquet'
    annotations_df = pd.read_parquet(annotation_file)
    print(annotations_df)

    # TODO: merge with more EEG_1 files
    data_file = '/Users/verenaojeda/RICSE/notebooks/sample_data/chb01_seizure_EEGwindow_1.npz'
    data = np.load(data_file, allow_pickle=True)
    print(list(data.keys()))
    print(data['EEG_win'].shape)
    print(data['EEG_win'].dtype)

    # # Close the file
    # data.close()

    # ## SAMPLE OF A NORMAL WINDOW
    # # Extract the data for the first window (we know from the annotations file this is a normal window)
    # normal_window_data = data['EEG_win'][0]
    # print(normal_window_data.shape)
    # show_window(normal_window_data)

    # ## SAMPLE OF A SEIZURE WINDOW
    # # Extract the data for the window 26524 (we know from the annotations file this is a seizure)
    # seizure_window_data = data['EEG_win'][26524]
    # print(seizure_window_data.shape)
    # show_window(seizure_window_data)

    # Extract labels from the DataFrame
    labels = annotations_df['class'].values
    data_ = data['EEG_win']
    data_float32 = data_.astype(np.float32)
    print(data_float32.shape)

    # Reshape data into (samples, features) format
    # where features = (channels * time_steps)
    n_samples = data_float32.shape[0]
    data_reshaped = data_float32.reshape(n_samples, -1)

    print(data_reshaped.shape)
    # Split the data into training and testing sets
    X_train, X_test, _, _ = train_test_split(data_reshaped, np.zeros(n_samples), test_size=0.2, random_state=42)

    # Reshape the data back to the original shape
    X_train = X_train.reshape(-1, 21, 128)
    X_test = X_test.reshape(-1, 21, 128)

    # Now you can split the corresponding labels if you have them, for example:
    # Assuming you have labels stored in a variable called 'labels'
    y_train, y_test, _, _ = train_test_split(labels, np.zeros(n_samples), test_size=0.2, random_state=42)

    # Your training and testing sets are ready
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    print(X_train[0])
    print(y_train[0])

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Convert data to DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_size = X_train.shape[2]
    hidden_size = 64
    output_size = 1

    model = EpilepsyLSTM(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labelss in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labelss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss}")

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        outputs = model(X_test_tensor)
        predicted = (outputs >= 0.5).squeeze().cpu().numpy()
        accuracy = (predicted == y_test).mean()

    print(f"Test Accuracy: {accuracy}")

    # # Define the number of folds for cross-validation
    # num_folds = 5
    # kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # # Initialize lists to store accuracy for each fold
    # accuracies = []

    # print(data_reshaped.shape)

    # # Perform k-fold cross-validation
    # for fold, (train_indices, val_indices) in enumerate(kf.split(data_reshaped)):
    #     print(f"Fold {fold + 1}/{num_folds}")

    #     print(train_indices)
    #     print(val_indices)
    #     # Split data into training and validation sets for this fold
    #     X_train_fold = data_reshaped[train_indices]
    #     X_val_fold = data_reshaped[val_indices]

    #     X_train_fold = X_train_fold.reshape(-1, 21, 128)
    #     X_val_fold = X_val_fold.reshape(-1, 21, 128)

    #     y_train_fold = labels[train_indices]
    #     y_val_fold = labels[val_indices]

    #     # Your training and testing sets are ready
    #     print(X_train_fold.shape)
    #     print(y_train_fold.shape)
    #     print(X_val_fold.shape)
    #     print(y_val_fold.shape)

    #     # Define model
    #     model = EpilepsyLSTM(input_size, hidden_size, output_size)

    #     # Define loss function and optimizer
    #     criterion = nn.BCELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

    #     X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
    #     y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32)
    #     X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
    #     # y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32)

    #     # Convert data to DataLoader
    #     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    #     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    #     # Training the model
    #     num_epochs = 10
    #     for epoch in range(num_epochs):
    #         model.train()
    #         running_loss = 0.0
    #         for inputs, labelss in train_loader:
    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = criterion(outputs.squeeze(), labelss)
    #             loss.backward()
    #             optimizer.step()
    #             running_loss += loss.item()

    #         print(f"Epoch {epoch+1}, Loss: {running_loss}")

    #     # Evaluate the model on validation set
    #     with torch.no_grad():
    #         model.eval()
    #         outputs = model(X_val_tensor)
    #         predicted = (outputs >= 0.5).squeeze().cpu().numpy()
    #         accuracy = (predicted == y_val_fold).mean()
    #         accuracies.append(accuracy)
    #         print(f"Validation Accuracy: {accuracy}")

    # # Calculate and print average accuracy across all folds
    # average_accuracy = np.mean(accuracies)
    # print(f"Average Validation Accuracy: {average_accuracy}")


if __name__ == '__main__':
    main()


