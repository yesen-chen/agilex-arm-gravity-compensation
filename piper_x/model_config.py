"""Shared PiperX dynamics-model configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPER_X_DYNAMICS_URDF = (
    PROJECT_ROOT / "piper_x" / "urdf" / "piper_x_with_gripper_tcp.urdf"
)
PIPER_X_TCP_FRAME = "tcp_link"
PIPER_X_ARM_DOFS = 6


def require_dynamics_urdf() -> str:
    """Return the generated URDF path or explain how to create it."""
    if not PIPER_X_DYNAMICS_URDF.is_file():
        generator = PROJECT_ROOT / "piper_x" / "build_gripper_dynamics_urdf.py"
        raise FileNotFoundError(
            f"缺少 PiperX 带夹爪动力学模型: {PIPER_X_DYNAMICS_URDF}\n"
            f"请先运行: python3 {generator}"
        )
    return str(PIPER_X_DYNAMICS_URDF)
