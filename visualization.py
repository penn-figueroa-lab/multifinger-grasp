import argparse
import sys
import os
import os.path as op
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import torch
import time
from scipy.spatial.transform import Rotation as R
import trimesh



def multiply(parent_pos, parent_ori, child_pos, child_ori):
    parent_rotation = R.from_quat(parent_ori)
    rotated_child_pos = parent_rotation.apply(child_pos)
    combined_pos = (
        parent_pos[0] + rotated_child_pos[0],
        parent_pos[1] + rotated_child_pos[1],
        parent_pos[2] + rotated_child_pos[2]
    )
    return combined_pos

def get_revolute_joint_positions(client, robot_id, joint_indices):
    joint_positions = []
    for joint_index in joint_indices:
        joint_info = client.getJointInfo(robot_id, joint_index)
        # print("joint_info", joint_info)
        parent_link_index = joint_info[16]
        # print("parent_link_index", parent_link_index)
        joint_frame_pos = joint_info[14]
        joint_frame_ori = joint_info[15] 
        # print("joint_frame_pos", joint_frame_pos)
        
        if parent_link_index == -1:
            # If the parent link index is -1, this means the joint is directly attached to the base link
            parent_world_pos, parent_world_ori = client.getBasePositionAndOrientation(robot_id)
        else:
            # Get the parent link state
            parent_world_pos, parent_world_ori, _, _, _, _ = client.getLinkState(robot_id, parent_link_index)
        
        joint_world_pos = multiply(parent_world_pos, parent_world_ori, joint_frame_pos, joint_frame_ori)
        joint_positions.append(joint_world_pos)
        # print(joint_world_pos)
    return joint_positions

def axis_angle_to_quaternion(axis_angle):
    axis = axis_angle[:3]
    angle = np.linalg.norm(axis)
    if angle < 1e-6:
        return [0, 0, 0, 1]
    axis_normalized = axis / angle
    return p.getQuaternionFromAxisAngle(axis_normalized, angle)

def reset_joint_states(client, robot_id, joint_positions, joint_indices):
    for joint_index, joint_value in zip(joint_indices, joint_positions):
        if isinstance(joint_value, torch.Tensor):
            joint_value = joint_value.item()
        client.resetJointState(robot_id, joint_index, joint_value)  # Reset the joint angle for each joint

def load_and_visualize(client, urdf_file, object_urdf, data, revolute_joint_indices, viz_revolute_joint_indices, revolute_joint_names, file_name, save_dir):
    """
    Load robot and object URDFs, and visualize the trajectory based on the loaded data.
    
    Parameters:
    - client (pybullet.BulletClient): The PyBullet client instance.
    - urdf_file (str): Path to the robot's URDF file.
    - object_urdf (str): Path to the object's URDF file.
    - data (dict): Loaded trajectory data from the .npy file.
    - revolute_joint_indices (list of int): Indices of the revolute joints to control.
    - revolute_joint_names (list of str): Names of the revolute joints.
    - file_name (str): Name of the current .npy file being visualized.
    """
    hand_data = data['right_hand']
    object_key = list(data.keys())
    object_key.remove('right_hand')
    print("object key", object_key)
    object_key = object_key[0]  # Assuming only one object per file
    object_data = data[object_key]
    
    # Extract data arrays
    trans_data = hand_data['trans']     # Shape: (batch, num_frames, 3)
    rot_data = hand_data['rot']         # Shape: (batch, num_frames, 3)
    pose_data = hand_data['pose']       # Shape: (batch, num_frames, num_joints)
    
    obj_trans_data = object_data['trans']  # Shape: (batch, num_frames, 3)
    obj_rot_data = object_data['rot']   # Shape: (batch, num_frames, 3)
    
    batch_size = trans_data.shape[0]
    for batch in range(batch_size):
        print(f"Visualizing Batch {batch + 1}/{batch_size} for file {file_name}...")
        
        num_frames = trans_data.shape[1]
        
        # Load robot URDF
        robot_id = client.loadURDF(
            urdf_file,
            basePosition=trans_data[batch, 0],
            baseOrientation=axis_angle_to_quaternion(rot_data[batch, 0]),
            useFixedBase=True  # Set to False if the robot should be dynamic
        )
        
        # Load object URDF
        object_id = client.loadURDF(
            object_urdf,
            basePosition=obj_trans_data[batch, 0],
            baseOrientation=axis_angle_to_quaternion(obj_rot_data[batch, 0]),
            useFixedBase=True
        )

        visual_shape_info = client.getVisualShapeData(object_id)
        n_points = 150
        link_point_cloud = {}
        for info in visual_shape_info:
            link_id = info[1]
            scale = info[3]
            mesh_file = info[4].decode()  # Mesh file path
            position = np.array(info[5])  # Link position
            orientation = np.array(info[6])  # Link orientation (quaternion)
            
            R_ = np.array(client.getMatrixFromQuaternion(orientation)).reshape((3, 3))
            mesh = trimesh.load(mesh_file, force='mesh')
            point_cloud, _ = trimesh.sample.sample_surface(mesh, n_points)
            point_cloud = point_cloud * np.array(scale)
            # print("R, positions", R_, position)
            point_cloud = point_cloud @ R_.T + position # point cloud in local frame
            link_point_cloud[link_id] = {"point_cloud": point_cloud} 
        
        joint_positions_over_time = np.zeros((num_frames, 3, len(viz_revolute_joint_indices)))
        point_clouds_over_time = np.zeros((num_frames, 3, n_points * 2))


        # Visualization loop for each frame
        print("Starting visualization loop...")
        for frame in range(num_frames):
            hand_trans = trans_data[batch, frame]
            hand_rot = rot_data[batch, frame]
    
            hand_quat = axis_angle_to_quaternion(hand_rot)
            client.resetBasePositionAndOrientation(robot_id, hand_trans, hand_quat)
    
            obj_trans = obj_trans_data[batch, frame]
            obj_rot = obj_rot_data[batch, frame]
            obj_quat = axis_angle_to_quaternion(obj_rot)
            client.resetBasePositionAndOrientation(object_id, obj_trans, obj_quat)

            link_world_point_cloud = {link_id: {"point_cloud": None} for link_id in link_point_cloud.keys()}
            R_world = R.from_quat(obj_quat).as_matrix()

            for link_id, link_data in link_point_cloud.items():
                ori_point_cloud = link_data["point_cloud"]
                point_cloud = ori_point_cloud @ R_world.T + obj_trans
                link_world_point_cloud[link_id]["point_cloud"] = point_cloud

            all_points = np.concatenate([link_data["point_cloud"] for link_data in link_world_point_cloud.values()], axis=0)
            point_clouds_over_time[frame, :, :] = all_points.T 

            # Extract and update revolute joint angles
            # Adjust the slicing based on how many revolute joints you have
            pose = pose_data[batch, frame][6:]  # Assuming you want to skip the first 6 joints
            reset_joint_states(client, robot_id, pose, revolute_joint_indices)
            joint_positions = get_revolute_joint_positions(client, robot_id, viz_revolute_joint_indices)
            # print("joint_positions", np.array(joint_positions).shape)

            joint_positions_over_time[frame, :, :] = np.array(joint_positions).T
            # Step simulation
            client.stepSimulation()
    
            # Control playback speed
            # time.sleep(0.02)  # Adjust as needed (e.g., 0.05 for ~20 FPS)
    
            # Optional: Print progress
            if frame % 10 == 0:
                print(f"Batch {batch + 1}, Frame {frame + 1}/{num_frames}")
    
        print(f"Completed visualization for Batch {batch + 1}/{batch_size} in file {file_name}.")
        # print("joint_positions over time", joint_positions_over_time)
        print("Completed point cloud over time", point_clouds_over_time.shape)

        # output_file = op.join(save_dir, f"joint_positions_{file_name}.npy")
        # np.save(output_file, joint_positions_over_time)

        # output_file = op.join(save_dir, f"point_cloud_{file_name}.npy")
        # np.save(output_file, point_clouds_over_time)
        
        # Optional: Remove robot and object to reset the simulation for the next batch
        client.removeBody(robot_id)
        client.removeBody(object_id)

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PyBullet Multi-File Robot Visualization")
    # parser.add_argument("--data_dir", type=str, default="/home/yihsuan/graspxl/raisim/GraspXL/raisimGymTorch/data_all/my_shadow_diverse_seq_npy", help="Path to the directory containing .npy files")
    parser.add_argument("--data_dir", type=str, default="/home/yihsuan/graspxl/raisim/GraspXL/raisimGymTorch/data_all/shadow_rotated_180_npy", help="Path to the directory containing .npy files")

    # parser.add_argument("--data_dir", type=str, default="/home/yihsuan/traj_gen_with_mano/diverse_seq_npy", help="Path to the directory containing .npy files")
    parser.add_argument("--urdf_file", type=str, default="/home/yihsuan/traj_gen_with_mano/shadow/shadowhand_large.urdf", help="Path to the robot's URDF file")
    parser.add_argument("--object_urdf", type=str, default="/home/yihsuan/traj_gen_with_mano/object_urdf/WineGlass_461664fa3a9ad7a18ee45bd8e008284e/WineGlass_461664fa3a9ad7a18ee45bd8e008284e_fixed_base.urdf", help="Path to the object's URDF file")
    parser.add_argument("--file_prefix", type=str, default="shadow_WineGlass_461664fa3a9ad7a18ee45bd8e008284e_", help="Prefix of the .npy files")
    parser.add_argument("--file_suffix", type=str, default=".npy", help="Suffix of the .npy files")
    parser.add_argument("--num_files", type=int, default=100, help="Number of .npy files to visualize")
    parser.add_argument("--save_dir", type=str, default="/home/yihsuan/traj_gen_with_mano/object_point_clouds", help="Directory to save output files")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    data_dir = args.data_dir
    urdf_file = args.urdf_file
    object_urdf = args.object_urdf
    file_prefix = args.file_prefix
    file_suffix = args.file_suffix
    num_files = args.num_files
    save_dir = args.save_dir

    if not op.exists(save_dir):
        os.makedirs(save_dir)
    
    # Initialize PyBullet
    client = bc.BulletClient(connection_mode=p.GUI)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())  # For default URDFs
    # client.setGravity(0, 0, -9.81)
    # Prepare list of .npy files
    npy_files = [f"{file_prefix}{i:03d}{file_suffix}" for i in range(0, num_files)]
    print(npy_files)
    # Identify Revolute Joints (Assuming they are consistent across files)
    # Load a sample file to extract joint indices
    sample_file_path = op.join(data_dir, npy_files[0])
    if not op.exists(sample_file_path):
        print(f"Sample file {sample_file_path} does not exist. Please check the path and filename.")
        return
    
    sample_data = np.load(sample_file_path, allow_pickle=True).item()
    sample_hand_data = sample_data['right_hand']
    
    # Temporarily load robot to identify revolute joints
    temp_robot_id = client.loadURDF(
        urdf_file,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True
    )
    
    revolute_joint_indices = []
    revolute_joint_names = []
    num_joints = client.getNumJoints(temp_robot_id)
    for joint_index in range(num_joints):
        joint_info = client.getJointInfo(temp_robot_id, joint_index)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joint_indices.append(joint_index)
            revolute_joint_names.append(joint_name)
    
    # Remove the temporary robot
    client.removeBody(temp_robot_id)
    
    # Optionally, exclude certain joints if necessary
    # For example, skip the first 3 revolute joints
    revolute_joint_indices = revolute_joint_indices[3:]
    revolute_joint_names = revolute_joint_names[3:]
    viz_revolute_joint_indices = [8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 25, 27, 28, 29, 30, 33, 34, 35, 36, 37]
    
    print("Identified Revolute Joint Indices:", revolute_joint_indices)
    print("Revolute Joint Names:", revolute_joint_names)
    print("Viz revolute joint indices", viz_revolute_joint_indices)
    
    # Iterate over each .npy file and visualize
    for npy_file in npy_files:
        file_path = op.join(data_dir, npy_file)
        if not op.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue
        
        print(f"\nLoading data from {npy_file}...")
        data = np.load(file_path, allow_pickle=True).item()
        
        # Load and visualize the data
        load_and_visualize(
            client=client,
            urdf_file=urdf_file,
            object_urdf=object_urdf,
            data=data,
            revolute_joint_indices=revolute_joint_indices,
            viz_revolute_joint_indices=viz_revolute_joint_indices,
            revolute_joint_names=revolute_joint_names,
            file_name=npy_file,
            save_dir=save_dir
        )
    
    print("All visualizations completed.")
    while True:
        client.stepSimulation()

    # client.disconnect()

if __name__ == "__main__":
    main()
