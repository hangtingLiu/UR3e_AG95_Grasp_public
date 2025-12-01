import os
import sys
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import spatialmath as sm

from manipulator_grasp.path_plan.set_plan import getIk, get_traj

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

_net = None
# ==================== 网络加载 ====================
def get_net():
    """
    加载训练好的 GraspNet 模型
    """
    global _net
    if _net is None:
        _net = GraspNet(input_feature_dim=0, 
                        num_view=300, 
                        num_angle=12, 
                        num_depth=4,
                        cylinder_radius=0.05, 
                        hmin=-0.02, 
                        hmax_list=[0.01, 0.02, 0.03, 0.04], 
                        is_training=False)
    _net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load('./logs/log_rs/checkpoint-rs.tar') # checkpoint_path
    _net.load_state_dict(checkpoint['model_state_dict'])
    _net.eval()
    return _net



# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # print("\n=== 尺寸验证 ===")
    # print("深度图尺寸:", depth.shape)
    # print("颜色图尺寸:", color.shape[:2])
    # print("工作空间尺寸:", workspace_mask.shape)

    # 构造相机内参矩阵
    height = color.shape[0]
    width = color.shape[1]
    fovy = np.pi / 4 # 定义的仿真相机
    focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
    c_x = width / 2.0   # 水平中心
    c_y = height / 2.0  # 垂直中心
    intrinsic = np.array([
        [focal, 0.0, c_x],    
        [0.0, focal, c_y],   
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0  # 深度因子，根据实际数据调整

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # mask = depth < 2.0
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

    NUM_POINT = 6000 # 10000或5000
    # 如果点数足够，随机采样NUM_POINT个点（不重复）
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs] # 提取点云和颜色

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    # end_points = {'point_clouds': cloud_sampled}

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d



# ==================== 主函数：获取抓取预测 ====================
def run_grasp_inference(color_path, depth_path, sam_mask_path=None):
    # 1. 加载网络
    net = get_net()

    # 2. 处理数据，此处使用返回的工作空间掩码路径
    end_points, cloud_o3d = get_and_process_data(color_path, depth_path, sam_mask_path)

    # 3. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 释放内存
    del end_points
    torch.cuda.empty_cache()
    
    # 4. 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 5. 碰撞检测
    COLLISION_THRESH = 0.01
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        collision_thresh = 0.01
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_o3d.points), voxel_size=voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]

    # 6. NMS 去重 + 按照得分排序（降序）
    gg.nms().sort_by_score()

    # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
    angle_threshold = np.deg2rad(45)  # 30度的弧度值
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第一列
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    else:
        print(f"\n[DEBUG] Filtered {len(filtered)} grasps within ±45° of vertical out of {len(all_grasps)} total predictions.")

    # ===== 新增部分：计算物体中心点 =====
    # 使用点云计算物体的中心点
    points = np.asarray(cloud_o3d.points)
    object_center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    # 计算每个抓取位姿中心点与物体中心点的距离
    distances = []
    for grasp in filtered:
        grasp_center = grasp.translation
        distance = np.linalg.norm(grasp_center - object_center)
        distances.append(distance)

    # 创建一个新的排序列表，包含距离和抓取对象
    grasp_with_distances = [(g, d) for g, d in zip(filtered, distances)]
    
    # 按距离升序排序（距离越小越好）
    grasp_with_distances.sort(key=lambda x: x[1])
    
    # 提取排序后的抓取列表
    filtered = [g for g, d in grasp_with_distances]

    # ===== 新增部分：综合得分和距离进行最终排序 =====
    # 创建一个新的排序列表，包含综合得分和抓取对象
    # 综合得分 = 抓取得分 * 0.7 + (1 - 距离/最大距离) * 0.3
    max_distance = max(distances) if distances else 1.0
    grasp_with_composite_scores = []

    for g, d in grasp_with_distances:
        # 归一化距离分数（距离越小分数越高）
        distance_score = 1 - (d / max_distance)
        
        # 综合得分 = 抓取得分 * 权重1 + 距离得分 * 权重2
        composite_score = g.score * 0.4 + distance_score * 0.6
        # print(f"\n g.score = {g.score}, distance_score = {distance_score}")
        grasp_with_composite_scores.append((g, composite_score))

    # 按综合得分降序排序
    grasp_with_composite_scores.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的抓取列表
    filtered_gg_list = [g for g, score in grasp_with_composite_scores]

    return filtered_gg_list, cloud_o3d


# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg_list, cloud_o3d):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。
    """

    # 0.初始准备阶段
    # 目标：计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
    n_wc = np.array([0.0, -1.0, 0.0])   # 相机朝向
    o_wc = np.array([-1.0, 0.0, -0.2])  # 相机朝向
    t_wc = np.array([1.2, 0.8, 1.6])    # 相机的位置。与scene.xml中保持一致。

    n_wp = np.array([0.0, 1.0, 0.0])   # 夹爪放置朝向 （与相机朝向绕z轴翻转180度）
    o_wp = np.array([1.0, 0.0, -0.5])  # 夹爪放置朝向
    t_wp = np.array([0.65, 0.2, 0.9])  # 夹爪放置位置

    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))

    T_wp = sm.SE3.Trans(t_wp) * sm.SE3(sm.SO3.TwoVectors(x=n_wp, y=o_wp))

    action = np.zeros(7)

    success_flag = False

    for grasp in gg_list:
        gg = GraspGroup()  # 创建临时容器
        gg.add(grasp)

        T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 1], y=gg.rotation_matrices[0][:, 2]))
        T_wo = T_wc * T_co
        # print(f'T_wo: {T_wo}')
        T1 = T_wo * sm.SE3(0.0, 0.0, -0.15)
        # print(f'T1: {T1}')
        q_start1 = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_goal1 = getIk(env, q_start1, T1)

        if q_goal1 is not None:
            T2 = T_wo
            q_start2 = q_goal1
            q_goal2 = getIk(env, q_start2, T2)

            if q_goal2 is not None:
                T3 = T_wo * sm.SE3(0.0, 0.0, -0.15)
                q_start3 = q_goal2 
                q_goal3 = getIk(env, q_start3, T3)

                if q_goal3 is not None:
                    print("")
                    print("------q_goal: IK solution found------")
                    q_traj1 = get_traj(env, q_start1, q_goal1)
                    q_traj2 = get_traj(env, q_start2, q_goal2)
                    q_traj3 = get_traj(env, q_start3, q_goal3)
                    if q_traj1 is not None and q_traj2 is not None and q_traj3 is not None:
                        print("")
                        print("------Path plan Successful-----")
                        visual = True
                        if visual:
                            grippers = gg.to_open3d_geometry_list()
                            o3d.visualization.draw_geometries([cloud_o3d, *grippers])
                            success_flag = True
                            break
                    else:
                        print("Path plan Failed: try next gg")
                else:
                    print("q_goal3: IK solution Failed: try next gg")
            else:
                print("q_goal2: IK solution Failed: try next gg")
        else:
            print("q_goal1: IK solution Failed: try next gg")

    if success_flag:
        # 1.接近抓取位姿
        for i in range(q_traj1.shape[1]): 
            action[:6] = q_traj1[:6, i]
            env.step(action)

        # 2.执行抓取
        for i in range(q_traj2.shape[1]): 
            action[:6] = q_traj2[:6, i]
            env.step(action)
        for i in range(920):
            action[-1] += 0.001
            env.step(action)

        # 3.提起物体
        for i in range(q_traj3.shape[1]): 
            action[:6] = q_traj3[:6, i]
            env.step(action)

        for i in range(50): 
            env.step(action)

        T4 = T_wp
        q_start4 = q_goal3
        while True:
            q_goal4 = getIk(env, q_start4, T4)
            if q_goal4 is not None:
                q_traj4 = get_traj(env, q_start4, q_goal4)
                if q_traj4 is not None:
                    print("------q_traj4 plan successful------")
                    q_start5 = q_goal4
                    q_goal5  = q_start1
                    q_traj5 = get_traj(env, q_start5, q_goal5)
                    if q_traj5 is not None:
                        print("------q_traj5 plan successful------")
                        break
                    else:
                        print("------q_traj5 plan filed, try again------")
                else:
                    print("------q_traj4 plan filed, try again------")

        # 4.移动物体到目标放置位置，松开夹爪
        for i in range(q_traj4.shape[1]): 
            action[:6] = q_traj4[:6, i]
            env.step(action)
        for i in range(920):
            action[-1] -= 0.001
            env.step(action)

        for i in range(50): 
            env.step(action)

        # 5.回到初始位置
        for i in range(q_traj5.shape[1]): 
            action[:6] = q_traj5[:6, i]
            env.step(action)

        for i in range(50): 
            env.step(action)

        # while True:
        #     env.step(action)
    else:
        print("Grasp Failed")
        # while True:
        #     env.step(action)



