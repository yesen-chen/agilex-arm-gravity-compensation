from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_URDF = (
    PROJECT_ROOT
    / "third_party"
    / "agx_arm_urdf"
    / "piper_x"
    / "urdf"
    / "piper_x_description.urdf"
)
GRIPPER_XACRO = (
    PROJECT_ROOT
    / "third_party"
    / "agx_arm_urdf"
    / "piper_x"
    / "urdf"
    / "piper_x_with_gripper_description.xacro"
)
GENERATED_URDF = (
    PROJECT_ROOT / "piper_x" / "urdf" / "piper_x_with_gripper_tcp.urdf"
)


def _mass(root: ET.Element) -> float:
    return sum(
        float(mass.get("value"))
        for mass in root.findall("./link/inertial/mass")
    )


class PiperXUrdfBuildTest(unittest.TestCase):
    def test_generated_model_is_reproducible_and_six_dof(self) -> None:
        from piper_x.build_gripper_dynamics_urdf import build_model

        with tempfile.TemporaryDirectory() as temp_dir:
            rebuilt = Path(temp_dir) / "piper_x_with_gripper_tcp.urdf"
            build_model(
                ARM_URDF,
                GRIPPER_XACRO,
                rebuilt,
                tcp_xyz=[0.0, 0.0, 0.138],
                tcp_rpy=[0.0, 0.0, 0.0],
            )
            self.assertEqual(rebuilt.read_bytes(), GENERATED_URDF.read_bytes())

        root = ET.parse(GENERATED_URDF).getroot()
        movable_joints = [
            joint
            for joint in root.findall("joint")
            if joint.get("type") not in {"fixed", "floating"}
        ]
        self.assertEqual(
            [joint.get("name") for joint in movable_joints],
            [f"joint{i}" for i in range(1, 7)],
        )
        for joint_name in ("gripper", "gripper_joint1", "gripper_joint2"):
            joint = root.find(f"./joint[@name='{joint_name}']")
            self.assertIsNotNone(joint)
            self.assertEqual(joint.get("type"), "fixed")
        self.assertIsNotNone(root.find("./link[@name='tcp_link']"))
        self.assertEqual(root.findall(".//visual"), [])
        self.assertEqual(root.findall(".//collision"), [])

    def test_gripper_mass_and_tcp_offset(self) -> None:
        bare_root = ET.parse(ARM_URDF).getroot()
        generated_root = ET.parse(GENERATED_URDF).getroot()
        self.assertAlmostEqual(_mass(generated_root) - _mass(bare_root), 0.54, places=9)

        tcp_joint = generated_root.find("./joint[@name='gripper_tcp_joint']")
        self.assertIsNotNone(tcp_joint)
        origin = tcp_joint.find("origin")
        self.assertIsNotNone(origin)
        self.assertEqual(
            [float(value) for value in origin.get("xyz").split()],
            [0.0, 0.0, 0.138],
        )
        self.assertEqual(tcp_joint.find("parent").get("link"), "gripper_base")
        self.assertEqual(tcp_joint.find("child").get("link"), "tcp_link")


class PiperXPinocchioTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import numpy as np
            from core.agx_pinocchio import AgxPinocchio
        except ImportError as exc:
            self.skipTest(f"Pinocchio runtime dependencies unavailable: {exc}")
        self.np = np
        self.model = AgxPinocchio(
            GENERATED_URDF,
            expected_nq=6,
            expected_nv=6,
            required_frames=["tcp_link"],
        )

    def test_model_mass_fk_and_jacobian(self) -> None:
        q = self.np.zeros(6)
        position, rotation = self.model.forward_kinematics(q, "tcp_link")
        jacobian = self.model.jacobian(q, "tcp_link")

        self.assertEqual(position.shape, (3,))
        self.assertEqual(rotation.shape, (3, 3))
        self.assertEqual(jacobian.shape, (6, 6))
        self.assertTrue(self.np.all(self.np.isfinite(jacobian)))
        self.assertAlmostEqual(self.model.total_mass, 4.847, places=6)

    def test_dynamics_include_gripper_and_controller_output_is_finite(self) -> None:
        from controller.task_imp_controller import CartesianImpedanceController
        from core.agx_pinocchio import AgxPinocchio

        q = self.np.array([0.2, -0.5, 0.4, 0.3, -0.2, 0.1])
        v = self.np.zeros(6)
        bare = AgxPinocchio(ARM_URDF, expected_nq=6, expected_nv=6)

        bare_tau = bare.nonlinear_effects(q, v)
        gripper_tau = self.model.nonlinear_effects(q, v)
        self.assertEqual(gripper_tau.shape, (6,))
        self.assertTrue(self.np.all(self.np.isfinite(gripper_tau)))
        self.assertGreater(self.np.linalg.norm(gripper_tau - bare_tau), 1e-4)

        controller = CartesianImpedanceController(
            str(GENERATED_URDF),
            dofs=6,
            frame_name="tcp_link",
        )
        target_pos, target_rot = controller.pin_model.forward_kinematics(
            q, "tcp_link"
        )
        torque = controller.compute_cartesian_torque(
            desired_pos=target_pos,
            desired_ori=target_rot,
            q_cur=q,
            v_cur=v,
        )
        self.assertEqual(torque.shape, (6,))
        self.assertTrue(self.np.all(self.np.isfinite(torque)))

    def test_dimension_and_frame_validation_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected nq"):
            from core.agx_pinocchio import AgxPinocchio

            AgxPinocchio(GENERATED_URDF, expected_nq=7)
        with self.assertRaisesRegex(ValueError, "不存在 frame"):
            self.model.require_frame("missing_tcp")
        with self.assertRaisesRegex(ValueError, "q 维度"):
            self.model.forward_kinematics(self.np.zeros(5), "tcp_link")


if __name__ == "__main__":
    unittest.main()
