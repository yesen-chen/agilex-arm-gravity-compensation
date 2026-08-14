import os
import numpy as np
import pinocchio as pin
from typing import Iterable, Optional, Tuple


class helper:

    @staticmethod
    def as_vec(x: np.ndarray, dim: int, name: str) -> np.ndarray:
        """将输入转换为固定维度的一维向量。

        参数:
            x: 输入数据。
            dim: 目标维度。
            name: 参数名（用于报错提示）。

        返回:
            np.ndarray: 形状为 (dim,) 的浮点向量。

        异常:
            ValueError: 输入维度与 dim 不一致。
        """
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape != (dim,):
            raise ValueError(f"{name} 维度必须为 ({dim},)")
        return arr

    @staticmethod
    def as_rot(rot: np.ndarray, name: str = "rot") -> np.ndarray:
        """将输入转换为 3x3 旋转矩阵。"""
        arr = np.asarray(rot, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(f"{name} 维度必须为 (3, 3)")
        return arr

    @staticmethod
    def orientation_error_rotmat(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
        """姿态误差（旋转矩阵输入，旋转向量输出）。"""
        r_t = helper.as_rot(target_rot, "target_rot")
        r_c = helper.as_rot(current_rot, "current_rot")
        r_err = r_t @ r_c.T
        return np.asarray(pin.log3(r_err), dtype=float).reshape(3).copy()


class AgxPinocchio:

    # ===== 初始化与模型状态 =====
    def __init__(
        self,
        urdf_path=None,
        *,
        expected_nq: Optional[int] = None,
        expected_nv: Optional[int] = None,
        required_frames: Optional[Iterable[str]] = None,
    ):
        """初始化 Pinocchio 机器人模型。

        参数:
            urdf_path: URDF 文件路径。

        异常:
            ValueError: urdf_path 为空。
        """
        if urdf_path is None:
            raise ValueError("urdf_path 不能为空")
        urdf_path = os.path.abspath(os.fspath(urdf_path))
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
        # Controllers only need the kinematic/dynamic model. RobotWrapper's
        # BuildFromURDF also resolves visual/collision meshes, which makes a
        # valid dynamics URDF fail when ROS package:// resources are absent.
        self.robot = pin.RobotWrapper(pin.buildModelFromUrdf(urdf_path))
        self.robot.data = self.robot.model.createData()
        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv
        self.urdf_path = urdf_path
        self.frame_names = tuple(frame.name for frame in self.robot.model.frames)
        self._default_gravity = self.robot.model.gravity.linear.copy()

        if expected_nq is not None and self.nq != int(expected_nq):
            raise ValueError(
                f"模型位置自由度不匹配: expected nq={expected_nq}, actual nq={self.nq}, "
                f"urdf={urdf_path}"
            )
        if expected_nv is not None and self.nv != int(expected_nv):
            raise ValueError(
                f"模型速度自由度不匹配: expected nv={expected_nv}, actual nv={self.nv}, "
                f"urdf={urdf_path}"
            )
        for frame_name in required_frames or ():
            self.require_frame(frame_name)

    @property
    def total_mass(self) -> float:
        """返回 URDF 中所有刚体的总质量。

        固定基座的惯性会被 Pinocchio 聚合到 ``inertias[0]``，因此该项也必须
        计入；它不是额外的虚拟质量。
        """
        return float(sum(inertia.mass for inertia in self.robot.model.inertias))

    def has_frame(self, frame_name: str) -> bool:
        """检查模型是否包含指定 frame。"""
        return frame_name in self.frame_names

    def require_frame(self, frame_name: str) -> int:
        """返回 frame id；不存在时给出包含 URDF 路径的明确错误。"""
        if not self.has_frame(frame_name):
            raise ValueError(f"模型中不存在 frame={frame_name!r}: {self.urdf_path}")
        return int(self.robot.model.getFrameId(frame_name))

    def summary(self) -> str:
        """返回适合启动日志的模型摘要。"""
        return (
            f"URDF={self.urdf_path}, nq={self.nq}, nv={self.nv}, "
            f"mass={self.total_mass:.4f} kg"
        )

    def _to_model_state(self, q: np.ndarray, v: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """严格校验并转换外部关节状态。

        参数:
            q: 关节位置向量。
            v: 关节速度向量，可为 None。

        返回:
            Tuple[np.ndarray, Optional[np.ndarray]]: (q_full, v_full_or_none)。

        异常:
            ValueError: q 或 v 维度与模型自由度不一致。
        """
        q_full = np.asarray(q, dtype=float).reshape(-1)
        if q_full.shape != (self.nq,):
            raise ValueError(f"q 维度必须与模型 nq={self.nq} 一致，实际为 {q_full.shape}")
        q_full = q_full.copy()

        if v is None:
            return q_full, None

        v_full = np.asarray(v, dtype=float).reshape(-1)
        if v_full.shape != (self.nv,):
            raise ValueError(f"v 维度必须与模型 nv={self.nv} 一致，实际为 {v_full.shape}")
        v_full = v_full.copy()
        return q_full, v_full

    def _set_gravity_from_base_orientation(self, base_orientation: Optional[np.ndarray]) -> np.ndarray:
        """根据基座姿态更新重力方向。

        参数:
            base_orientation: 基座姿态旋转矩阵 (3, 3)；为 None 时使用默认重力方向。

        返回:
            np.ndarray: 更新前的重力向量（用于调用方恢复）。

        异常:
            ValueError: base_orientation 格式非法。

        原理:
            若 R_wb 表示 base 相对 world 的旋转，则重力在 base 坐标系下为
            g_b = R_wb^T @ g_w。动力学计算始终在模型基坐标进行，
            因此需要先把 world 重力旋转到 base。
        """
        old_gravity = self.robot.model.gravity.linear.copy()
        if base_orientation is None:
            self.robot.model.gravity.linear = self._default_gravity.copy()
            return old_gravity

        rot = helper.as_rot(base_orientation, "base_orientation")

        gravity_world = self._default_gravity
        gravity_base = rot.T @ gravity_world
        self.robot.model.gravity.linear = gravity_base
        return old_gravity

    # ===== 运动学基础 =====
    def frame_id(self, frame_name: str) -> int:
        """按名称获取 frame id。

        参数:
            frame_name: 模型中的 frame 名称。

        返回:
            int: frame 对应的 id。
        """
        return self.require_frame(frame_name)

    def forward_kinematics(
        self,
        q: np.ndarray,
        frame_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算指定 frame 的位姿。

        参数:
            q: 关节位置向量。
            frame_name: 目标 frame 名称。

        返回:
            Tuple[np.ndarray, np.ndarray]: (position[3], rotation_matrix[3,3])。
        """
        q_full, _ = self._to_model_state(q)
        fid = self.frame_id(frame_name)
        pin.forwardKinematics(self.robot.model, self.robot.data, q_full)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        placement = self.robot.data.oMf[fid]
        return placement.translation.copy(), placement.rotation.copy()

    def jacobian(
        self,
        q: np.ndarray,
        frame_name: str,
        reference: int = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    ) -> np.ndarray:
        """计算指定 frame 的 6xN 几何雅可比。

        参数:
            q: 关节位置向量。
            frame_name: 目标 frame 名称。
            reference: 雅可比参考系（Pinocchio ReferenceFrame）。

        返回:
            np.ndarray: 指定 frame 的 6xN 几何雅可比。
        """
        q_full, _ = self._to_model_state(q)
        fid = self.frame_id(frame_name)
        jac_full = pin.computeFrameJacobian(self.robot.model, self.robot.data, q_full, fid, reference)
        return jac_full.copy()

    # ===== 动力学核心 =====
    def nonlinear_effects(
        self,
        q: np.ndarray,
        v: np.ndarray,
        base_orientation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算非线性项 n(q, qd)。

        参数:
            q: 关节位置向量。
            v: 关节速度向量。
            base_orientation: 基座姿态旋转矩阵 (3, 3)。

        返回:
            np.ndarray: 非线性项 n(q, qd) = C(q, qd) @ qd + g(q)。

        原理:
            该项包含速度相关项与重力项，是控制中常见的前馈补偿目标。
        """
        q_full, v_full = self._to_model_state(q, v)
        old_gravity = self._set_gravity_from_base_orientation(base_orientation)
        try:
            nle = pin.nonLinearEffects(self.robot.model, self.robot.data, q_full, v_full)
        finally:
            self.robot.model.gravity.linear = old_gravity
        return nle.copy()

    def inverse_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray,
        base_orientation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算逆动力学力矩 tau = M(q) @ qdd + C(q, qd) @ qd + g(q)。

        参数:
            q: 关节位置向量。
            v: 关节速度向量 qd。
            a: 关节加速度向量 qdd。
            base_orientation: 基座姿态旋转矩阵 (3, 3)。

        返回:
            np.ndarray: 与输入 q 等长的逆动力学力矩。

        原理:
            使用 RNEA 直接计算完整动力学项。该接口是动力学总入口，
            其中重力补偿对应 v=0 且 a=0 的特例。
        """
        q_full, v_full = self._to_model_state(q, v)
        a_full = helper.as_vec(a, self.nv, "a")
        old_gravity = self._set_gravity_from_base_orientation(base_orientation)
        try:
            tau = pin.rnea(
                self.robot.model,
                self.robot.data,
                q_full,
                v_full,
                a_full,
            )
        finally:
            self.robot.model.gravity.linear = old_gravity
        return tau.copy()
