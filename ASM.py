import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # For progress bars
import open3d as o3d
from ADMM import run_admm, plot_errors

def normalize_mesh(mesh):
    # Extract vertices
    verts = mesh.verts_packed()
    center = verts.mean(0)
    mesh.offset_verts_(-center)
    verts = mesh.verts_packed()
    scale = verts.abs().max().item() 
    mesh.scale_verts_(1 / scale)
    return mesh, center, scale

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asm_data = np.load("ASM.npz", allow_pickle=True)
    X_train = asm_data["X_train"]
    X_test = asm_data["X_test"]
    y_train = asm_data["y_train"]
    y_test = asm_data["y_test"]
    idx_train = asm_data["idx_train"]
    idx_test = asm_data["idx_test"]
    
    print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")

    # Fit PCA on Training Data
    print("Fitting PCA on training data...")
    n_components = 50

    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    mean_shape = pca.mean_
    principal_components = pca.components_

    print("weights", X_train_pca.shape)
    print("mean_shape", mean_shape.shape)
    print("principal_componenet", principal_components.shape)

    print(f"PCA completed. Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"PCA fitted with {n_components} components.")

    # reconstruct(X_test, idx_test, y_test, mean_shape, principal_components, device)
    cs = (X_train - mean_shape) @ principal_components.T
    c_low = np.min(cs, axis=0)
    c_high = np.max(cs, axis=0)
    c_mean = np.mean(cs, axis=0)
    # print(np.sum(cs, axis=1))
    print(c_low)
    print(c_high)
    # print(c_mean)

    sample_idx = 1  # 0-19 
    F_obs = X_test[sample_idx]

    original_idx = idx_test[sample_idx]
    original_file = y_test[sample_idx]

    for i in range(len(X_test)):
        original_idx = idx_test[i]
        original_file = y_test[i]
        print(f"index {i}, original file {original_file}")

    target_mesh = load_objs_as_meshes([original_file], device=device)

    norm_mesh, center, scale = normalize_mesh(target_mesh)
    scale *= 0.01
    print("center and scale", center, scale)

    print(F_obs.shape)
    print(mean_shape.shape)
    print(principal_components.shape)
    c_gt = principal_components @ (F_obs - mean_shape)
    Yt = F_obs.reshape((-1, 3)).T

    R_temp = np.random.randn(3, 3)
    U, _, Vt = np.linalg.svd(R_temp)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = np.random.randn(3) * 100
    Yt_rotated = scale * R @ Yt + t[:, None]
    temp_t = np.mean(Yt_rotated, axis=1, keepdims=True)
    Yt_rotated -= temp_t
    # Yt_rotated = Yt

    B0 = mean_shape.reshape((-1, 3)).T
    B = np.linalg.pinv(principal_components).reshape(-1, 3, n_components).swapaxes(0,1)

    # print("F_obs", Yt.shape)
    # print("mean shape", B0.shape)
    # print("PC", B.shape)

    # # Visualize the point cloud
    print("GT shape", c_gt)
    print("GT Scale", scale)
    print("GT Rotation", R)
    print("GT_translation", t)

    Xt, R_pred, s_pred, t_pred, c_pred, errors = run_admm(B0, B, Yt_rotated, c_gt, c_low, c_high, c_mean, pca.explained_variance_ratio_, scale, R, t, temp_t, Yt, rho=0.1, iterations=10)
    print("R_pred", R_pred)
    print("s_pred", s_pred)
    print("t_pred", t_pred + temp_t[:, 0])
    print("c_pred", c_pred)

    # plot_errors(errors)
    Xt = Xt.T
    Yt = Yt.T
    Yt_rotated = Yt_rotated.T

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(scale * Yt)
    green_color = np.array([0, 1, 0])  
    pcd1.colors = o3d.utility.Vector3dVector(np.tile(green_color, (Yt.shape[0], 1)))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(Yt_rotated)
    red_color = np.array([1, 0, 0])  
    pcd2.colors = o3d.utility.Vector3dVector(np.tile(red_color, (Yt_rotated.shape[0], 1)))

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(Xt)
    blue_color = np.array([0, 0, 1])  
    pcd3.colors = o3d.utility.Vector3dVector(np.tile(blue_color, (Xt.shape[0], 1)))

    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    # print('Xt', Xt[:, :5])
    # print("Yt", Yt[:, :5])

    # difference = np.mean(np.linalg.norm(Yt - Xt, axis=0))
    # print(f"difference between Yt and Xt: {difference:.6f}" )

    # f = (F_obs - mean_shape) @ principal_components.T
    # print("c latent", f)


if __name__ == "__main__":
    main()