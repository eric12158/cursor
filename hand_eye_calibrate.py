"""
手眼标定（通用版，可配置输入）

目标：
- 不再依赖 MATLAB 工具箱导出的 cameraParams.mat
- 相机内参/畸变、标定板参数、图片与机械臂位姿，都从可编辑的配置/数据文件输入
- 标定板识别与 solvePnP 的流程，参考 `import sys.py` 的做法（圆点/棋盘格两种）
- 手眼算法核心仍保留：使用 OpenCV `cv2.calibrateHandEye`

使用方式（示例）：
    python hand_eye_calibrate.py --config hand_eye_config.json

输出：
- `cam2end.txt`（默认，可在配置里改名）：4x4 齐次矩阵 T_camera_to_gripper（相机到末端/法兰）
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from math import cos, pi, sin
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray  # 3x3
    dist: np.ndarray  # (N,)


@dataclass(frozen=True)
class TargetConfig:
    type: str  # "circles" | "chessboard"
    rows: int
    cols: int
    spacing_mm: float


@dataclass(frozen=True)
class DatasetConfig:
    images_dir: str
    image_glob: str
    poses_file: str
    pairing: str  # "sorted" | "by_name"
    poses_format: str  # "txt_list" | "json"


@dataclass(frozen=True)
class RobotPoseConfig:
    pose_is: str  # "base_to_gripper" | "gripper_to_base"
    rotation_unit: str  # "deg" | "rad"
    euler_order: str  # only supports "xyz" for now


@dataclass(frozen=True)
class OutputConfig:
    cam2end_txt: str


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_config(path: str) -> Tuple[CameraModel, TargetConfig, DatasetConfig, RobotPoseConfig, OutputConfig, int]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _require("camera" in cfg, "config 缺少 camera")
    _require("target" in cfg, "config 缺少 target")
    _require("dataset" in cfg, "config 缺少 dataset")

    cam = cfg["camera"]
    K = np.array(cam["K"], dtype=float)
    dist = np.array(cam.get("dist", [0, 0, 0, 0, 0]), dtype=float).reshape(-1)
    _require(K.shape == (3, 3), f"camera.K 形状应为 3x3，实际为 {K.shape}")

    target = cfg["target"]
    tgt = TargetConfig(
        type=str(target.get("type", "circles")).strip().lower(),
        rows=int(target["rows"]),
        cols=int(target["cols"]),
        spacing_mm=float(target["spacing_mm"]),
    )
    _require(tgt.type in {"circles", "chessboard"}, "target.type 仅支持 circles 或 chessboard")
    _require(tgt.rows > 0 and tgt.cols > 0, "target.rows/cols 必须 > 0")
    _require(tgt.spacing_mm > 0, "target.spacing_mm 必须 > 0")

    d = cfg["dataset"]
    ds = DatasetConfig(
        images_dir=str(d["images_dir"]),
        image_glob=str(d.get("image_glob", "*.*")),
        poses_file=str(d["poses_file"]),
        pairing=str(d.get("pairing", "sorted")).strip().lower(),
        poses_format=str(d.get("poses_format", "txt_list")).strip().lower(),
    )
    _require(ds.pairing in {"sorted", "by_name"}, "dataset.pairing 仅支持 sorted 或 by_name")
    _require(ds.poses_format in {"txt_list", "json"}, "dataset.poses_format 仅支持 txt_list 或 json")

    r = cfg.get("robot_pose", {})
    rp = RobotPoseConfig(
        pose_is=str(r.get("pose_is", "base_to_gripper")).strip().lower(),
        rotation_unit=str(r.get("rotation_unit", "deg")).strip().lower(),
        euler_order=str(r.get("euler_order", "xyz")).strip().lower(),
    )
    _require(rp.pose_is in {"base_to_gripper", "gripper_to_base"}, "robot_pose.pose_is 仅支持 base_to_gripper 或 gripper_to_base")
    _require(rp.rotation_unit in {"deg", "rad"}, "robot_pose.rotation_unit 仅支持 deg 或 rad")
    _require(rp.euler_order == "xyz", "robot_pose.euler_order 目前仅支持 xyz")

    o = cfg.get("output", {})
    out = OutputConfig(cam2end_txt=str(o.get("cam2end_txt", "cam2end.txt")))

    method_name = str(cfg.get("hand_eye", {}).get("method", "TSAI")).strip().upper()
    method_map: Dict[str, int] = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    _require(method_name in method_map, f"hand_eye.method 不支持: {method_name}，可选: {sorted(method_map.keys())}")
    method = method_map[method_name]

    return CameraModel(K=K, dist=dist), tgt, ds, rp, out, method


def list_images(images_dir: str, image_glob: str) -> List[str]:
    import glob

    pattern = os.path.join(images_dir, image_glob)
    files = glob.glob(pattern)
    files.sort()
    return files


def _parse_pose_line(line: str) -> Optional[List[float]]:
    s = line.strip()
    if not s:
        return None
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    # 支持逗号或空格分隔
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) < 6:
        return None
    vals = [float(x) for x in parts[:6]]
    return vals


def load_robot_poses_txt_list(path: str) -> List[List[float]]:
    poses: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = _parse_pose_line(line)
            if p is not None:
                poses.append(p)
    return poses


def load_robot_poses_txt_by_name(path: str) -> Dict[str, List[float]]:
    """
    TXT 按文件名配对格式（每行）：
      img_0001.jpg, x, y, z, rx, ry, rz
    或：
      img_0001.jpg x y z rx ry rz
    也允许写完整路径，程序会取 basename 作为 key。
    """
    out: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
            parts = [p for p in s.replace(",", " ").split() if p]
            if len(parts) < 7:
                continue
            name = os.path.basename(parts[0])
            try:
                vals = [float(x) for x in parts[1:7]]
            except ValueError:
                continue
            out[name] = vals
    return out


def load_robot_poses_json(path: str) -> Dict[str, List[float]]:
    """
    JSON 格式示例：
    {
      "img_0001.jpg": [x,y,z,rx,ry,rz],
      "img_0002.jpg": [x,y,z,rx,ry,rz]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _require(isinstance(data, dict), "poses_format=json 时，poses_file 必须是 {filename: [x,y,z,rx,ry,rz]} 的字典")
    out: Dict[str, List[float]] = {}
    for k, v in data.items():
        _require(isinstance(v, (list, tuple)) and len(v) >= 6, f"{k} 的位姿应为长度>=6的数组")
        out[str(k)] = [float(x) for x in v[:6]]
    return out


def myRPY2R_robot(x: float, y: float, z: float) -> np.ndarray:
    """欧拉角 xyz（roll/pitch/yaw）-> 旋转矩阵，矩阵形式与 `import sys.py` 的 xyz 顺序保持一致。"""
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]], dtype=float)
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]], dtype=float)
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def pose_to_RT_base_to_gripper(pose: Sequence[float], rotation_unit: str) -> np.ndarray:
    _require(len(pose) >= 6, "pose 长度不足 6")
    x, y, z, rx, ry, rz = [float(v) for v in pose[:6]]
    if rotation_unit == "deg":
        rx, ry, rz = rx * pi / 180.0, ry * pi / 180.0, rz * pi / 180.0
    R = myRPY2R_robot(rx, ry, rz)
    t = np.array([[x], [y], [z]], dtype=float)
    RT = np.column_stack([R, t])
    RT = np.row_stack((RT, np.array([0, 0, 0, 1], dtype=float)))
    return RT


def invert_RT(RT: np.ndarray) -> np.ndarray:
    _require(RT.shape == (4, 4), "RT 必须为 4x4")
    R = RT[:3, :3]
    t = RT[:3, 3:4]
    RTi = np.eye(4, dtype=float)
    RTi[:3, :3] = R.T
    RTi[:3, 3:4] = -R.T @ t
    return RTi


def make_object_points(target: TargetConfig) -> np.ndarray:
    rows, cols, sp = target.rows, target.cols, target.spacing_mm
    if target.type == "chessboard":
        # 与 import sys.py 保持一致：输入 rows/cols 是“总行列”，内角点为 (rows-1)x(cols-1)
        objp = np.zeros(((rows - 1) * (cols - 1), 3), np.float32)
        objp[:, :2] = np.mgrid[0 : cols - 1, 0 : rows - 1].T.reshape(-1, 2) * sp
    else:
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sp
    return objp


def detect_target_points(image_bgr: np.ndarray, target: TargetConfig) -> Tuple[bool, Optional[np.ndarray]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rows, cols = target.rows, target.cols
    if target.type == "chessboard":
        ret, corners = cv2.findChessboardCorners(
            gray,
            (cols - 1, rows - 1),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners if ret else None
    else:
        ret, corners = cv2.findCirclesGrid(gray, (cols, rows), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        return ret, corners if ret else None


def solve_target_pose(objp: np.ndarray, corners: np.ndarray, cam: CameraModel) -> Tuple[np.ndarray, np.ndarray]:
    ok, rvec, tvec = cv2.solvePnP(objp, corners, cam.K, cam.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    _require(bool(ok), "solvePnP 失败")
    R, _ = cv2.Rodrigues(rvec)
    t = np.array(tvec, dtype=float).reshape(3, 1)
    return R, t


def save_matrix_txt(path: str, RT: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in RT:
            f.write(" ".join(map(str, row.tolist())) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="配置文件路径（JSON）")
    parser.add_argument("--strict", action="store_true", help="遇到任意一张图识别失败就直接报错退出")
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    cam, target, ds, rp, out, method = load_config(args.config)

    image_paths = list_images(ds.images_dir, ds.image_glob)
    _require(len(image_paths) > 0, f"未找到图片：{ds.images_dir}/{ds.image_glob}")

    # 加载机械臂位姿
    if ds.poses_format == "json":
        pose_map = load_robot_poses_json(ds.poses_file)
        poses_list: List[Tuple[str, List[float]]] = []
        for p in image_paths:
            name = os.path.basename(p)
            if name not in pose_map:
                if args.strict:
                    raise ValueError(f"poses_file 缺少图片 {name} 的位姿")
                continue
            poses_list.append((p, pose_map[name]))
    else:
        if ds.pairing == "by_name":
            pose_map = load_robot_poses_txt_by_name(ds.poses_file)
            poses_list = []
            for p in image_paths:
                name = os.path.basename(p)
                if name not in pose_map:
                    if args.strict:
                        raise ValueError(f"poses_file 缺少图片 {name} 的位姿")
                    continue
                poses_list.append((p, pose_map[name]))
        else:
            poses = load_robot_poses_txt_list(ds.poses_file)
            _require(
                len(poses) >= len(image_paths),
                f"位姿数量不足：poses={len(poses)} images={len(image_paths)}（请检查 dataset.poses_file 或改用 poses_format=json）",
            )
            poses_list = list(zip(image_paths, poses[: len(image_paths)]))

    objp = make_object_points(target)

    R_gripper2base: List[np.ndarray] = []
    t_gripper2base: List[np.ndarray] = []
    R_target2cam: List[np.ndarray] = []
    t_target2cam: List[np.ndarray] = []

    used = 0
    for img_path, pose in poses_list:
        img = cv2.imread(img_path)
        if img is None:
            if args.strict:
                raise ValueError(f"无法读取图片：{img_path}")
            continue

        found, corners = detect_target_points(img, target)
        if not found or corners is None:
            if args.strict:
                raise ValueError(f"标定板识别失败：{img_path}")
            continue

        R_tc, t_tc = solve_target_pose(objp, corners, cam)
        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)

        # 机械臂位姿：默认认为输入的是 T_base_to_gripper（与 import sys.py 一致）
        RT_base_to_gripper = pose_to_RT_base_to_gripper(pose, rotation_unit=rp.rotation_unit)
        if rp.pose_is == "base_to_gripper":
            RT_gripper_to_base = invert_RT(RT_base_to_gripper)
        else:
            RT_gripper_to_base = RT_base_to_gripper

        R_gripper2base.append(RT_gripper_to_base[:3, :3])
        t_gripper2base.append(RT_gripper_to_base[:3, 3:4])
        used += 1

    _require(used >= 3, f"有效数据不足（仅 {used} 组），至少需要 3 组以上才能稳定手眼标定")
    _require(len(R_gripper2base) == len(R_target2cam), "数据配对异常：机械臂位姿与图像姿态数量不一致")

    # === 手眼标定（算法保留：cv2.calibrateHandEye）===
    R_cam_to_gripper, t_cam_to_gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method,
    )

    print("手眼矩阵分解得到的旋转矩阵")
    print(R_cam_to_gripper)
    print("\n")
    print("手眼矩阵分解得到的平移矩阵")
    print(t_cam_to_gripper)

    RT_cam_to_gripper = np.column_stack((R_cam_to_gripper, t_cam_to_gripper))
    RT_cam_to_gripper = np.row_stack((RT_cam_to_gripper, np.array([0, 0, 0, 1], dtype=float)))

    print("\n相机相对于末端(法兰)的变换矩阵为：")
    print(RT_cam_to_gripper)

    save_matrix_txt(out.cam2end_txt, RT_cam_to_gripper)

    # === 结果验证：每一组推回去的 base->target 应该接近一致 ===
    # 计算：T_gripper_to_base * T_camera_to_gripper * T_target_to_camera = T_target_to_base
    # 再求逆得到 T_base_to_target（更直观）
    for i in range(len(R_target2cam)):
        RT_gripper_to_base = np.eye(4, dtype=float)
        RT_gripper_to_base[:3, :3] = R_gripper2base[i]
        RT_gripper_to_base[:3, 3:4] = t_gripper2base[i]

        RT_target_to_cam = np.eye(4, dtype=float)
        RT_target_to_cam[:3, :3] = R_target2cam[i]
        RT_target_to_cam[:3, 3:4] = t_target2cam[i]

        RT_target_to_base = RT_gripper_to_base @ RT_cam_to_gripper @ RT_target_to_cam
        RT_base_to_target = invert_RT(RT_target_to_base)

        print(f"第 {i} 次 (base->target):")
        print(RT_base_to_target[:3, :])
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
