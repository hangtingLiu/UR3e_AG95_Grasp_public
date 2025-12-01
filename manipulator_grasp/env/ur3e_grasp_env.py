import os.path
import sys

sys.path.append('../../manipulator_grasp')

import numpy as np
import mujoco
import mujoco.viewer

import glfw
import cv2


from manipulator_grasp.path_plan.set_model import (
    load_models,
    add_self_collisions,
    add_object_collisions,
    load_path_planner,
)


class UR3eGraspEnv:

    def __init__(self):
        self.sim_hz = 500

        self.model: mujoco.MjModel = None
        self.data: mujoco.MjData = None

        self.model_roboplan = None
        self.collision_model = None
        self.data_roboplan = None
        self.target_frame = None
        self.ik = None
        self.rrt_options = None

        self.renderer: mujoco.Renderer = None
        self.depth_renderer: mujoco.Renderer = None
        self.viewer: mujoco.viewer.Handle = None

        self.height = 640 # 256 640 720
        self.width = 640 # 256 640 1280
        self.fovy = np.pi / 4

        # 新增离屏渲染相关属性
        self.camera_name = "cam"
        self.camera_id = -1
        self.offscreen_context = None
        self.offscreen_scene = None
        self.offscreen_camera = None
        self.offscreen_viewport = None
        self.glfw_window = None

    def reset(self):
        # 初始化路径规划模型
        urdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robot_description', 'urdf', 'ur3e_ag95.urdf')
        srdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robot_description', 'srdf', 'ur3e_ag95.srdf')
        self.model_roboplan, self.collision_model, visual_model = load_models(urdf_path)
        add_self_collisions(self.model_roboplan, self.collision_model, srdf_path)
        add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.04)

        self.data_roboplan = self.model_roboplan.createData()

        self.target_frame, self.ik, self.rrt_options = load_path_planner(self.model_roboplan, self.data_roboplan, self.collision_model)
        
        # 初始化 MuJoCo 模型和数据
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(filename)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        self.data.ctrl[:6] = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

        # 创建两个渲染器实例，分别用于生成彩色图像和深度图
        self.renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        self.depth_renderer = mujoco.renderer.Renderer(self.model, height=self.height, width=self.width)
        # 更新渲染器中的场景数据
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        # 启用深度渲染
        self.depth_renderer.enable_depth_rendering()
        
        # 初始化被动查看器
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # 为了方便观察
        self.viewer.cam.lookat[:] = [1.8, 1.1, 1.7]  # 对应XML中的center
        self.viewer.cam.azimuth = 210      # 对应XML中的azimuth
        self.viewer.cam.elevation = -35    # 对应XML中的elevation
        self.viewer.cam.distance = 1.2     # 根据场景调整的距离值
        self.viewer.sync() # 立即同步更新

        # # --- 新增: 初始化离屏渲染 ---
        # # 初始化GLFW用于离屏渲染
        # glfw.init()
        # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        # self.glfw_window = glfw.create_window(self.width, self.height, "Offscreen", None, None)
        # glfw.make_context_current(self.glfw_window)

        # # 获取相机ID
        # self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        # if self.camera_id != -1:
        #     print(f"成功找到相机 '{self.camera_name}', ID: {self.camera_id}")
        #     # 使用XML中定义的固定相机
        #     self.offscreen_camera = mujoco.MjvCamera()
        #     mujoco.mjv_defaultCamera(self.offscreen_camera)
        #     self.offscreen_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        #     self.offscreen_camera.fixedcamid = self.camera_id

        # # 创建离屏场景和上下文
        # self.offscreen_scene = mujoco.MjvScene(self.model, maxgeom=10000)
        # self.offscreen_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        # self.offscreen_viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        # mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.offscreen_context)

        # # 创建OpenCV窗口
        # cv2.namedWindow('MuJoCo Camera Output', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('MuJoCo Camera Output', self.width, self.height)


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
        if self.depth_renderer is not None:
            self.depth_renderer.close()

        # 清理离屏渲染资源
        cv2.destroyAllWindows()
        if self.glfw_window is not None:
            glfw.destroy_window(self.glfw_window)
        glfw.terminate()
        self.offscreen_context = None
        self.offscreen_scene = None

    def step(self, action=None):
        if action is not None:
            self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        self.viewer.sync()

        # # --- 新增: 离屏渲染和显示 ---
        # if all([self.offscreen_context, self.offscreen_scene, self.offscreen_camera]):
        #     # 更新场景
        #     mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), 
        #                          mujoco.MjvPerturb(), self.offscreen_camera, 
        #                          mujoco.mjtCatBit.mjCAT_ALL.value, self.offscreen_scene)
            
        #     # 渲染到离屏缓冲区
        #     mujoco.mjr_render(self.offscreen_viewport, self.offscreen_scene, self.offscreen_context)
            
        #     # 读取像素数据
        #     rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        #     mujoco.mjr_readPixels(rgb, None, self.offscreen_viewport, self.offscreen_context)
            
        #     # 转换颜色空间并显示
        #     bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        #     cv2.imshow('MuJoCo Camera Output', bgr)
            
        #     # 检查ESC键
        #     if cv2.waitKey(1) == 27:
        #         print("用户按下了ESC键,退出仿真。")
        #         self.close()
        #         exit(0)
                
    def render(self):
        '''
        常用于强化学习或机器人控制任务中，提供环境的视觉观测数据。
        '''
        # 更新渲染器中的场景数据
        self.renderer.update_scene(self.data, 0)
        self.depth_renderer.update_scene(self.data, 0)
        # 渲染图像和深度图
        return {
            'img': self.renderer.render(),
            'depth': self.depth_renderer.render()
        }



if __name__ == '__main__':
    env = UR3eGraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
