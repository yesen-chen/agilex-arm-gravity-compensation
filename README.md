# 动力学简单应用

## 安装匹诺曹库

```bash
sudo apt install ros-$ROS_DISTRO-pinocchio ros-$ROS_DISTRO-hpp-fcl ros-$ROS_DISTRO-coal
```

## 激活CAN模块

```bash
bash can_activate.sh
```

## 安装SDK

运行前需先安装新版 SDK：  
[agilexrobotics/pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)

## 项目结构

- `core/`：Pinocchio 封装、URDF/MDH 工具与解析层识别
- `controller/`：关节阻抗与笛卡尔阻抗控制器
- `nero/`：Nero 机型示例脚本
- `piper/`：Piper 机型示例脚本
- `piper_x/`：PiperX 机型示例脚本

## 运行示例

### Piper

```bash
python3 piper/main_gc.py             # 重力补偿
python3 piper/main_jnt_imp.py        # 关节阻抗
python3 piper/main_tast_imp.py       # 笛卡尔阻抗
```

### PiperX

```bash
python3 piper_x/main_gc.py
python3 piper_x/main_jnt_imp.py
python3 piper_x/main_tast_imp.py
```

### Nero

```bash
python3 nero/main_gc.py
python3 nero/main_jnt_imp.py
python3 nero/main_tast_imp.py
```

## PiperX 带夹爪动力学模型

PiperX 的三个控制示例使用
`piper_x/urdf/piper_x_with_gripper_tcp.urdf`。该模型由官方
`third_party/agx_arm_urdf` 数据生成，包含 flange、夹爪基座和双指的质量与
惯量，但仍保持机械臂模型为 6-DOF。夹爪开合不由阻抗控制器控制。

### 生成模型与标定 TCP

默认 TCP 位于 `gripper_base` 坐标系的 `[0, 0, 0.138]` 米：

```bash
git clone https://github.com/agilexrobotics/agx_arm_urdf.git \
  third_party/agx_arm_urdf
python3 piper_x/build_gripper_dynamics_urdf.py
```

仓库已经提交生成后的轻量 URDF，运行控制器不需要克隆官方 URDF。只有重新生成
模型或修改 TCP 时才需要上述官方仓库；`third_party/agx_arm_urdf/` 默认被
`.gitignore` 排除，不会进入本仓库。

若实测 TCP 偏移为其他数值，请重新生成模型。例如：

```bash
python3 piper_x/build_gripper_dynamics_urdf.py \
  --tcp-xyz 0.0 0.0 0.145 \
  --tcp-rpy 0.0 0.0 0.0
```

生成器会锁定夹爪的模型关节（保留惯性），并移除动力学不需要的网格，因此
加载模型不依赖 ROS package 路径。修改上游 URDF 或 TCP 后需要重新运行生成器。

### 离线验证

建议为本项目创建独立的 uv 环境。PyPI 包名为 `pin`，Python 导入名为
`pinocchio`：

```bash
cd /home/descfly/yesen/third_party/agilex-arm-gravity-compensation
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python numpy scipy pin

.venv/bin/python -c "import numpy, pinocchio; print(pinocchio.__version__)"
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -t . -v
```

运行真机脚本还需要按照官方说明安装 `pyAgxArm`。

测试会检查：

- 模型仅包含六个活动机械臂关节；
- `tcp_link` 存在且 Jacobian 为 `6x6`；
- flange 和夹爪增加约 `0.54 kg`；
- 带夹爪与裸臂的重力补偿力矩存在合理差异；
- 阻抗控制输出为有限的 6 维关节力矩。

### 真机验证顺序

1. 先加载模型，确认启动日志显示 `nq=6`、`nv=6`、质量约 `4.847 kg`。
2. 不启用笛卡尔刚度，离线记录多个姿态下裸臂/带夹爪重力力矩差异。
3. 机械臂悬空，从设备允许范围内的保守力矩限幅测试 `piper_x/main_gc.py`。
4. 确认重力补偿方向正确后，再以低于示例默认值的 `K/B` 启动
   `piper_x/main_tast_imp.py`。
5. TCP 标定未完成、末端接触环境或夹持未知载荷时，不应直接使用高刚度。

控制器本身不会替代驱动器的力矩、速度和关节限位保护。抓取物体后若需要精确
补偿，还必须将物体质量、质心和惯量加入负载模型。
