#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agx_pinocchio import AgxPinocchio
from piper_x.model_config import PIPER_X_ARM_DOFS, require_dynamics_urdf

def main():
    # 6-DOF 动力学模型包含固定夹爪惯性；夹爪开合仍由独立通道控制。
    urdf_path = require_dynamics_urdf()

    # 初始化动力学模型（用于重力/非线性项补偿计算）
    pin = AgxPinocchio(
        urdf_path,
        expected_nq=PIPER_X_ARM_DOFS,
        expected_nv=PIPER_X_ARM_DOFS,
    )
    print(f"Pinocchio 模型: {pin.summary()}")

    # 控制频率
    control_frequency = 200.0

    # 初始化机械臂接口
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=PiperFW.V188,
        channel="can0",
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    if robot.joint_nums != PIPER_X_ARM_DOFS:
        raise RuntimeError(
            f"PiperX 关节数应为 {PIPER_X_ARM_DOFS}，实际为 {robot.joint_nums}"
        )

    # 等待机械臂使能
    while not robot.enable():
        time.sleep(0.01)
    print("机械臂使能成功")
    
    # 获取当前关节角度（等待首帧有效数据）
    joint_angles = None
    while joint_angles is None:
        js = robot.get_joint_angles()
        if js is not None:
            joint_angles = np.array(js.msg)
            break
        time.sleep(0.01)

    # 计算世界坐标系到基座坐标系的旋转矩阵
    roll, pitch, yaw = 0, 0, 0  # 单位：deg
    R_world_base = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    print("开始重力补偿控制循环...")
    
    try:
        while True:
            start_time = time.time()

            # 获取当前关节角度和速度
            joint_angles = np.array(robot.get_joint_angles().msg)

            joint_velocities = np.zeros(robot.joint_nums)
            for i in range(1, robot.joint_nums + 1):
                ms = robot.get_motor_states(i)
                if ms is not None:
                    joint_velocities[i - 1] = ms.msg.velocity

            # 计算重力补偿扭矩
            gravity_torque = pin.inverse_dynamics(
                joint_angles,
                joint_velocities,
                np.zeros_like(joint_velocities),
                R_world_base,
            )

            # 应用重力补偿扭矩
            try: 
                for joint_id in range(1, robot.joint_nums + 1):
                    robot.move_mit(joint_id, 0, 0, 0, 0, gravity_torque[joint_id - 1])
                    
            except Exception as e:
                print(f"应用重力补偿失败: {e}")
            
            # 控制频率
            t = 1.0 / control_frequency
            elapsed_time = time.time() - start_time
            if elapsed_time < t:
                time.sleep(t - elapsed_time)
            else:
                print(f"警告：控制循环超时 {elapsed_time:.3f}s > {t:.3f}s")
                
    except KeyboardInterrupt:
        print("\n用户中断，停止重力补偿")
        for joint_id in range(1, robot.joint_nums + 1):
            try:
                robot.move_mit(joint_id, joint_angles[joint_id - 1], 0, 10, 0.8, 0)
            except:
                pass
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        for joint_id in range(1, robot.joint_nums + 1):
            try:
                robot.move_mit(joint_id, joint_angles[joint_id - 1], 0, 10, 0.8, 0)
            except:
                pass


if __name__ == "__main__":
    main()
    