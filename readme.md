# UR3e_AG95_Grasp_Public

## 基础环境配置

请参考以下仓库的环境配置：[VLM_Grasp_Interactive](https://github.com/hangtingLiu/VLM_Grasp_Interactive)

## 仓库说明

与另一个仓库相比，该仓库（first commit版本）对代码进行了以下调整：

1. **移除多模态交互部分**：
   - 注释了未使用的多模态交互部分。
   - 直接通过鼠标点击目标，结合 SAM 分割和 GraspNet 提供目标夹爪位姿。

2. **新增避障运动规划部分**：
   - 需要安装以下两个库：
     - [pyroboplan](https://github.com/your-link)
     - [pinocchio](https://github.com/your-link)

## 注意事项

当前版本代码在轨迹规划时未加入被夹取物体的避障部分。如果需要更完善的功能，可以参考 pyroboplan 库自行实现。实现思路如下：

- 根据库中的函数，加入末端物体的碰撞体。

---

> **提示**: 如果有任何问题，请随时提交 Issue 或联系维护者。