import pytorch_kinematics as pk
import trimesh
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import torch
import torch.optim as optim
import os
import csv
import raisimpy
import time
from scipy.spatial.transform import Rotation as R


# def get_revolute_joint_positions(client, robot_id, joint_indices):
#     """
#     Get the current 3D positions of the revolute joints.

#     """
#     joint_positions = []
#     print('joint_indices', joint_indices)
#     for joint_index in joint_indices:
#         link_state = client.getLinkState(robot_id, joint_index)
#         print("link_state", link_state)
#         joint_position = link_state[0]  # The first element is the 3D world position
#         joint_positions.append(joint_position)
#     return joint_positions

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
        # print("parent_world_pos", parent_world_pos)
        # print("parent_world_ori", parent_world_ori)
        # print("joint_frame_pos", joint_frame_pos)
        # print("joint_frame_ori", joint_frame_ori)

        # Transform the joint relative position to the world frame
        # joint_world_pos = client.multiplyTransforms(
        #     parent_world_pos, parent_world_ori, joint_frame_pos, joint_frame_ori
        # )[0]  # Extract only the position
        joint_world_pos = multiply(parent_world_pos, parent_world_ori, joint_frame_pos, joint_frame_ori)
        joint_positions.append(joint_world_pos)
        # print(joint_world_pos)
    return joint_positions

def visualize_joints(client, robot_id, joint_positions, color = [0, 0, 1], sphere_radius=0.05):
    link_joints = {}
    link_joints[-1] = np.array([client.getBasePositionAndOrientation(robot_id)[0]])
    for idx, joint_position in enumerate(joint_positions):
        joint_position_array = np.array([joint_position])
        link_joints[idx] = joint_position_array
    print("Drawing link point clouds")
    client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 1)
    for key, world_point_cloud in link_joints.items():
        # print("world_point_cloud", world_point_cloud)
        point_cloud_id = client.addUserDebugPoints(world_point_cloud, np.zeros_like(world_point_cloud), pointSize=10)
        link_joints[key] = {"point_cloud": world_point_cloud, "point_cloud_id": point_cloud_id}
    client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 1)

    line_ids = []
    newlist = [3, 7, 11, 16]
    connect_to_base = [0, 4, 8, 12, 17]
    for i in range(len(joint_positions) - 1):
        if i not in newlist:
            line_id = client.addUserDebugLine(
                joint_positions[i], 
                joint_positions[i + 1], 
                lineColorRGB=[1, 0, 0], 
                lineWidth=1
            )
            line_ids.append(line_id)
        if i in connect_to_base:
            line_id = client.addUserDebugLine(
                link_joints[-1]["point_cloud"][0], 
                joint_positions[i], 
                lineColorRGB=[1, 0, 0], 
                lineWidth=1
            )
            line_ids.append(line_id)
    
    client.configureDebugVisualizer(client.COV_ENABLE_RENDERING, 1)

    return link_joints, line_ids

def update_visualization(client, robot_id, joint_positions, link_joints, line_ids):
    for key in link_joints.keys():
        if key == -1:
            new_joint_position = np.array([client.getBasePositionAndOrientation(robot_id)[0]])
        else:
            new_joint_position = np.array([joint_positions[key]])
        point_cloud_id = client.addUserDebugPoints(new_joint_position, np.zeros_like(new_joint_position), pointSize=10,
                                                   replaceItemUniqueId=link_joints[key]["point_cloud_id"])
        link_joints[key]["point_cloud"] = new_joint_position
        link_joints[key]["point_cloud_id"] = point_cloud_id

    line_index = 0
    newlist = [3, 7, 11, 16]
    connect_to_base = [0, 4, 8, 12, 17]
    for i in range(len(joint_positions) - 1):
        if i not in newlist:
            # Update line between consecutive joints
            line_id = client.addUserDebugLine(
                joint_positions[i], 
                joint_positions[i + 1], 
                lineColorRGB=[1, 0, 0], 
                lineWidth=1,
                replaceItemUniqueId=line_ids[line_index]
            )
            line_index += 1

        if i in connect_to_base:
            # Update line connecting joint to the base
            line_id = client.addUserDebugLine(
                link_joints[-1]["point_cloud"][0], 
                joint_positions[i], 
                lineColorRGB=[1, 0, 0], 
                lineWidth=1,
                replaceItemUniqueId=line_ids[line_index]
            )
            line_index += 1

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "/home/yihsuan/traj_gen_with_mano/diverse_seq_npy"
    npy_file = "shadow_WineGlass_3.npy"
    file_path = os.path.join(data_dir, npy_file)

    data = np.load(file_path, allow_pickle=True)    
    data = data.item()
    hand_data = data['right_hand']
    object_data = data['WineGlass_461664fa3a9ad7a18ee45bd8e008284e']

    print("hand_trans", hand_data['trans']) # world pose
    print("hand_rot", hand_data['rot'].shape) # axis-angle
    print("hand_pose", hand_data['pose'].shape) # joint angles

    print("Loading PyBullet and the robot model")
    client = bc.BulletClient(connection_mode=p.GUI)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())  # Used for loading URDFs and other assets

    urdf_file = "/home/yihsuan/traj_gen_with_mano/shadow/shadowhand_large.urdf"

    trans_data = hand_data['trans'][0, :, :]
    rot_data = hand_data['rot'][0, :, :]
    pose_data = hand_data['pose'][0, :, :]
    num_frames = trans_data.shape[0]

    obj_trans_data = object_data['trans'][0, :, :]
    obj_rot_data = object_data['rot'][0, :, :]

    robot_id = client.loadURDF(urdf_file,
                               basePosition=trans_data[0],
                               baseOrientation=axis_angle_to_quaternion(rot_data[0]),
                               useFixedBase=True
                               )
    
    # robot_id = client.loadURDF(urdf_file,
    #                            basePosition=trans_data[0],
    #                            useFixedBase=True
    #                            )
    
    
    revolute_joint_indices = []
    revolute_joint_names = []
    num_joints = client.getNumJoints(robot_id)
    for joint_index in range(num_joints):
        joint_info = client.getJointInfo(robot_id, joint_index)
        print("joint_info", joint_info)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        # print(f"Joint {joint_index}: {joint_name}")
        # print(f"  Type: {joint_type}")
        # print(f"  Limits: [{lower_limit}, {upper_limit}]")
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joint_indices.append(joint_index)
            revolute_joint_names.append(joint_name)
            # print(f"Revolute Joint {joint_index}: {joint_name}")
            # print(f"  Limits: [{lower_limit}, {upper_limit}]")

    revolute_joint_indices = revolute_joint_indices[3:]
    # viz_revolute_joint_indices = [x + 1 for x in revolute_joint_indices]
    viz_revolute_joint_indices = [8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 25, 27, 28, 29, 30, 33, 34, 35, 36, 37]
    print(viz_revolute_joint_indices)
    revolute_joint_names = revolute_joint_names[3:]
    
    joint_positions = get_revolute_joint_positions(client, robot_id, viz_revolute_joint_indices)

    print("revolute_joint_indices", revolute_joint_indices)
    print("revolute_joint_names", revolute_joint_names)
    print("joint positions", joint_positions)

    link_joints, line_ids = visualize_joints(client, robot_id, joint_positions)

    obj_urdf = "/home/yihsuan/traj_gen_with_mano/object_urdf/WineGlass_461664fa3a9ad7a18ee45bd8e008284e/WineGlass_461664fa3a9ad7a18ee45bd8e008284e_fixed_base.urdf"
    object_id = client.loadURDF(obj_urdf, 
                                basePosition= obj_trans_data[0],
                                baseOrientation=axis_angle_to_quaternion(obj_rot_data[0]),
                                useFixedBase=True)

    print("Starting visualization loop...")
    for frame in range(num_frames):
        joint_positions = get_revolute_joint_positions(client, robot_id, viz_revolute_joint_indices)
        # print("joint positions", len(joint_positions))
        update_visualization(client, robot_id, joint_positions, link_joints, line_ids)
        # print("length of joint_indices", link_joints)

        hand_trans = trans_data[frame]
        hand_rot = rot_data[frame]

        # time.sleep(1000)

        hand_quat = axis_angle_to_quaternion(hand_rot)
        client.resetBasePositionAndOrientation(robot_id, hand_trans, hand_quat)

        obj_trans = obj_trans_data[frame]
        obj_rot = obj_rot_data[frame]
        obj_quat = axis_angle_to_quaternion(obj_rot)
        client.resetBasePositionAndOrientation(object_id, obj_trans, obj_quat)

        pose = pose_data[frame][6:]
        reset_joint_states(client, robot_id, pose, revolute_joint_indices)
        joint_positions = get_revolute_joint_positions(client, robot_id, viz_revolute_joint_indices)

        update_visualization(client, robot_id, joint_positions, link_joints, line_ids)
        time.sleep(0.2)

        client.stepSimulation()


        if frame % 1 == 0:
            print(f"Frame {frame}/{num_frames}")

    print("Visualization completed.")
    # while True:
    #     client.stepSimulation()



