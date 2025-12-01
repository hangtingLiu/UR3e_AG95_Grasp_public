import os
import sys
import cv2
import mujoco
import matplotlib.pyplot as plt 
import time
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from manipulator_grasp.env.ur3e_grasp_env import UR3eGraspEnv

from vlm_process import segment_image
from grasp_process import run_grasp_inference, execute_grasp


# 全局变量
global color_img, depth_img, env
color_img = None
depth_img = None
env = None

# 获取彩色和深度图像数据
def get_image(env):
    global color_img, depth_img
     # 从环境渲染获取图像数据
    imgs = env.render()

    # 提取彩色和深度图像数据
    color_img = imgs['img']   # 这是RGB格式的图像数据
    depth_img = imgs['depth'] # 这是深度数据

    # 将RGB图像转换为OpenCV常用的BGR格式
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('color_img_path.jpg', color_img)

    return color_img, depth_img

# 构造回调函数，不断调用
def callback(color_frame, depth_frame):
    global color_img, depth_img
    scaling_factor_x = 1
    scaling_factor_y = 1

    color_img = cv2.resize(
        color_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_AREA
    )
    depth_img = cv2.resize(
        depth_frame, None,
        fx=scaling_factor_x,
        fy=scaling_factor_y,
        interpolation=cv2.INTER_NEAREST
    )

    if color_img is not None and depth_img is not None:
        test_grasp()


def test_grasp():
    global color_img, depth_img, env

    if color_img is None or depth_img is None:
        print("[WARNING] Waiting for image data...")
        return

    # 图像处理部分
    masks = segment_image(color_img) 

    # 完成后释放SAM和Whisper的内存
    torch.cuda.empty_cache()

    gg_list, cloud_o3d = run_grasp_inference(color_img, depth_img, masks)

    execute_grasp(env, gg_list, cloud_o3d)

    # 释放抓取模型的内存
    del gg_list
    torch.cuda.empty_cache()



if __name__ == '__main__':
    
    env = UR3eGraspEnv()
    env.reset()


    while True:

        for i in range(100): 
            env.step()

        color_img, depth_img = get_image(env)

        callback(color_img, depth_img)





    