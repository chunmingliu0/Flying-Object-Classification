import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt

# Path and label mapping
folder_path = '/content/drive/MyDrive/MPE_Research/Data/Train_new/'
keyword1 = 'SNR40'
keyword2 = 'Fz2'
all_data_list = []
all_labels_list = []

# Construct file_list: includes full paths of all .mat files
file_list = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if keyword1 in f and keyword2 in f and f.endswith('.mat') and os.path.isfile(os.path.join(folder_path, f))
]

for file_path in file_list:
    mat = scipy.io.loadmat(file_path)
    print(f"Contents of {file_path}: keys = {mat.keys()}")
    all_data_list.append(mat['all_data'])

    # Fix label format
    raw_labels = mat['all_labels']
    clean_labels = [str(l[0]) if isinstance(l, (np.ndarray, list)) else str(l) for l in raw_labels]
    all_labels_list.append(clean_labels)

# Merge data
all_data = np.vstack(all_data_list)
numeric_labels = np.concatenate(all_labels_list)

print(f"Merged data shape: {all_data.shape}")
print(f"Merged label shape: {numeric_labels.shape}")

# 2️⃣ Standardization (Z-score)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(all_data)

# 3️⃣ View original class distribution
counter = Counter(numeric_labels)
print(f"Original class distribution: {counter}")

# Visualize original class distribution
plt.bar(counter.keys(), counter.values())
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Original Class Distribution")
plt.show()

# Apply under-sampling with a fixed random_state for reproducibility
rus = RandomUnderSampler(random_state=42)

# Perform under-sampling (input: data_scaled and numeric_labels)
X_resampled, y_resampled = rus.fit_resample(data_scaled, numeric_labels)

print(f"\nResampled data shape: {X_resampled.shape}")
print(f"Resampled label shape: {y_resampled.shape}")

# View resampled class distribution
counter_resampled = Counter(y_resampled)
print("Resampled class distribution:", counter_resampled)

# Visualize resampled class distribution
plt.figure()
plt.bar(counter_resampled.keys(), counter_resampled.values())
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Resampled Class Distribution')
plt.show()

# Save as .npy files
from sklearn.model_selection import train_test_split

# Split into train, val, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save arrays
np.save('/content/drive/MyDrive/MPE_Research/Data/Train/X_train_Fz2_SNR40.npy', X_train)
np.save('/content/drive/MyDrive/MPE_Research/Data/Train/y_train_Fz2_SNR40.npy', y_train)
np.save('/content/drive/MyDrive/MPE_Research/Data/Train/X_val_Fz2_SNR40.npy', X_val)
np.save('/content/drive/MyDrive/MPE_Research/Data/Train/y_val_Fz2_SNR40.npy', y_val)
np.save('/content/drive/MyDrive/MPE_Research/Data/Test/X_test_Fz2_SNR40.npy', X_test)
np.save('/content/drive/MyDrive/MPE_Research/Data/Test/y_test_Fz2_SNR40.npy', y_test)
