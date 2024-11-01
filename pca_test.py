import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Define the data directory
data_dir = "/home/yihsuan/graspxl/raisim/GraspXL/raisimGymTorch/data_all/my_shadow_diverse_seq_npy"

# Define the filename pattern
npy_file_pattern = "shadow_WineGlass_461664fa3a9ad7a18ee45bd8e008284e_{:03d}.npy"

# Initialize lists to store combined data and labels
combined_data_list = []
trajectory_labels = []  # To keep track of which data point belongs to which trajectory
frame_labels = []       # To keep track of frame indices within trajectories

# Number of trajectories and frames per trajectory
start = 0
num_trajectories = 10
frames_per_trajectory = 200  # Assuming each trajectory has 200 frames

# Loop through all trajectories
for traj_idx in range(start, num_trajectories):
    # Construct the filename
    npy_file = npy_file_pattern.format(traj_idx)
    file_path = os.path.join(data_dir, npy_file)
    
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Warning: File {file_path} does not exist. Skipping.")
        continue
    
    # Load the data
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Skipping.")
        continue
    
    # Extract hand and object data
    hand_data = data.get('right_hand', {})
    object_data = data.get('WineGlass_461664fa3a9ad7a18ee45bd8e008284e', {})

    trans_data = hand_data['trans'][0, :, :]
    rot_data = hand_data['rot'][0, :, :]
    pose_data = hand_data['pose'][0, :, :][:, 6:]
    num_frames = trans_data.shape[0]

    obj_trans_data = object_data['trans'][0, :, :]
    obj_rot_data = object_data['rot'][0, :, :]
    
    combined = np.hstack((trans_data, rot_data, pose_data, obj_trans_data, obj_rot_data))  # Shape: (200, 31)
    
    # Append to the combined data list
    combined_data_list.append(combined)
    
    # Create labels
    trajectory_labels.extend([traj_idx] * frames_per_trajectory)
    frame_labels.extend(range(frames_per_trajectory))

# Convert the list to a NumPy array
if not combined_data_list:
    raise ValueError("No valid data found. Please check the data directory and file naming conventions.")

combined_data = np.vstack(combined_data_list)  # Shape: (num_valid_trajectories * 200, 31)
print("Combined data shape:", combined_data.shape)

# Optional: Create labels as NumPy arrays for further analysis
trajectory_labels = np.array(trajectory_labels)
frame_labels = np.array(frame_labels)

# Standardize the data
scaler = StandardScaler()
combined_data_std = scaler.fit_transform(combined_data)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(combined_data_std)
print("Principal components shape:", principal_components.shape)  # (num_valid_trajectories * 200, 2)

# Explained Variance
print("Explained variance by each component:", pca.explained_variance_ratio_)
print("Total explained variance:", np.sum(pca.explained_variance_ratio_))

# Visualization
plt.figure(figsize=(12, 10))

cmap = ListedColormap(plt.cm.tab20.colors[:num_trajectories])  # Select a palette with enough colors for each trajectory

for traj_id in range(start, num_trajectories):
    traj_points = principal_components[trajectory_labels == traj_id]
    traj_frames = frame_labels[trajectory_labels == traj_id]
    # by trajectory color
    # plt.scatter(traj_points[:, 0], traj_points[:, 1], 
    #             color=cmap(traj_id), marker='o', label=f'Trajectory {traj_id}')
    
    # by frame index
    plt.scatter(traj_points[:, 0], traj_points[:, 1], c = traj_frames,
                cmap='viridis', marker='o', label=f'Trajectory {traj_id}')

plt.title('PCA of Hand and Object Data Across 100 Trajectories')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Frame Index')
plt.grid(True)
plt.legend(title="Trajectory ID")
plt.tight_layout()
plt.show()
