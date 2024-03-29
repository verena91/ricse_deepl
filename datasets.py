import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class AutoencoderCustomDataset(Dataset):
    def __init__(self, csv_file=None, root_dir=None, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

        # Load CSV data
        self.data = pd.read_csv(csv_file)
        # Filter only rows with 'NEGATIVE' diagnosis
        self.data = self.data[self.data['DENSITAT'] == 'NEGATIVA']

        # Get list of patient codes with 'NEGATIVE' diagnosis
        self.patient_codes = self.data['CODI'].tolist()
        # print(self.patient_codes)

    def _load_patient_patches(self, patient_code):
        patches = []
        patient_dir = os.path.join(self.root_dir, patient_code)
        # print(patient_dir)
        if os.path.exists(patient_dir) and os.path.isdir(patient_dir):
            for filename in os.listdir(patient_dir):
                if filename.endswith(".png"):
                    patch_path = os.path.join(patient_dir, filename)
                    patches.append(patch_path)
        return patches

    def __len__(self):
        return sum(len(self._load_patient_patches(patient_code + '_1')) for patient_code in self.patient_codes)

    def __getitem__(self, idx):
        for patient_code in self.patient_codes:
            # print(patient_code + '_1')
            patches = self._load_patient_patches(patient_code + '_1')
            # print(patches)
            if idx < len(patches):
                img_name = patches[idx]
                # print(img_name)
                # image = cv2.imread(img_name)
                image = Image.open(img_name).convert('RGB')

                if image is not None:
                    if self.transform:
                        image = self.transform(image)

                    return image, 0  # Label 0 for NEGATIVE diagnosis
                else:
                    print('File not found')
                    return None, None

            idx -= len(patches)