"""
通用离线手眼标定（保留原有 OpenCV calibrateHandEye 算法调用）

目标：
- 不再依赖 MATLAB cameraParams.mat
- 相机内参/畸变/标定板参数/数据路径/机器人位姿单位与方向 -> 全部可配置（JSON）
- 图片与机械臂坐标组织方式参考 `import sys.py` 的思路（配置 + 统一数据结构）

使用：
  1) 准备配置文件（默认：hand_eye_config.json）
  2) 准备数据：
     - 推荐：calibration_data.json（每张图对应一组 robot_pose）
     - 兼容：images/pos.txt（与图片按排序索引一一对应）
  3) 运行：
     python hand_eye_calibrate.py --config hand_eye_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from dataclasses import dataclass
from math import cos, sin, pi
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ----------------------------
# 配置与数据结构
# ----------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        # 相机内参（必填）
        "fx": 0.0,
        "fy": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        # 畸变：k1,k2,p1,p2,k3（长度可为 4/5，缺省按 0 补齐）
        "dist": [0, 0, 0, 0, 0],
    },
    "calibration": {
        # 标定板类型：chessboard（棋盘格） / circles（圆点阵）
        "type": "chessboard",
        # rows/cols：参考 import sys.py 的约定
        # - chessboard：rows/cols 表示“格子数”，内角点数为 (rows-1) x (cols-1)
        # - circles：rows/cols 表示“圆点数”，点数为 rows x cols
        "rows": 7,
        "cols": 9,
        # spacing：mm，棋盘格为“方格边长”，圆点为“圆心间距”
        "spacing": 25.0,
        # 角点亚像素优化（建议 True）
        "refine_corners": True,
    },
    "dataset": {
        # 图片所在目录
        "images_dir": "./images",
        # 支持的图片后缀
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        # 推荐：显式数据文件（参考 import sys.py 的 calibs_data_list 思路）
        # 文件内容为 list，每项至少包含：
        #   {"img_path": "xxx.jpg", "robot_pose": [x,y,z,rx,ry,rz]}
        # img_path 可写相对路径（相对 images_dir 或 config 所在目录）
        "data_file": "./calibration_data.json",
        # 兼容：旧格式位姿文件（每行: [x,y,z,rx,ry,rz]）
        "pose_file": "./images/pos.txt",
        # 位姿角度单位：rad / deg
        "pose_angle_unit": "rad",
        # robot_pose 的含义：true 表示输入为 T_base_to_gripper；false 表示输入为 T_gripper_to_base
        # 注意：cv2.calibrateHandEye 需要的是 gripper->base（T_gripper_to_base）
        "pose_is_base_to_gripper": True,
    },
    "output": {
        "cam2end_path": "./cam2end.txt",
        "report_path": "./handeye_report.json",
    },
    # OpenCV 手眼标定方法：可选 "TSAI" / "PARK" / "HORAUD" / "ANDREFF" / "DANIILIDIS"
    # 留空则使用 OpenCV 默认
    "handeye_method": "",
}


@dataclass
class Sample:
    img_path: str
    robot_pose: Optional[List[float]]  # [x,y,z,rx,ry,rz]


# ----------------------------
# 基础工具
# ----------------------------

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        _deep_merge(cfg, user_cfg)
        return cfg
    # 若配置不存在：写出模板，方便用户直接编辑
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    raise FileNotFoundError(f"未找到配置文件，已生成模板：{path}（请填写相机内参后再运行）")


def euler_xyz_to_R(x: float, y: float, z: float) -> np.ndarray:
    """与参考脚本一致：xyz（外旋/固定轴） -> 等价于 R = Rz @ Ry @ Rx。"""
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]], dtype=float)
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]], dtype=float)
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def pose_to_T(pose: Sequence[float], angle_unit: str) -> np.ndarray:
    x, y, z, rx, ry, rz = [float(v) for v in pose]
    if angle_unit.lower() == "deg":
        rx = rx / 180.0 * pi
        ry = ry / 180.0 * pi
        rz = rz / 180.0 * pi
    Rm = euler_xyz_to_R(rx, ry, rz)
    t = np.array([[x], [y], [z]], dtype=float)
    T = np.vstack([np.hstack([Rm, t]), np.array([0, 0, 0, 1], dtype=float)])
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3:4]
    R_inv = Rm.T
    t_inv = -R_inv @ t
    return np.vstack([np.hstack([R_inv, t_inv]), np.array([0, 0, 0, 1], dtype=float)])


def read_pose_txt(file_path: str) -> List[List[float]]:
    """兼容旧 pos.txt：每行格式如 [x,y,z,rx,ry,rz] 或 x,y,z,rx,ry,rz"""
    data_list: List[List[float]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = line.replace("[", "").replace("]", "")
        values = [v.strip() for v in line.split(",") if v.strip() != ""]
        if len(values) < 6:
            raise ValueError(f"位姿行解析失败（至少6列）：{line}")
        data_list.append([float(v) for v in values[:6]])
    return data_list


def list_images(images_dir: str, exts: Sequence[str]) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    # 去重 + 排序，保证可复现
    files = sorted(list(dict.fromkeys(files)))
    return files


def resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    # 先按 base_dir 解析，再按当前工作目录兜底
    cand = os.path.normpath(os.path.join(base_dir, p))
    if os.path.exists(cand):
        return cand
    return os.path.normpath(os.path.join(os.getcwd(), p))


def load_samples(cfg: Dict[str, Any], config_path: str) -> List[Sample]:
    ds = cfg["dataset"]
    config_dir = os.path.dirname(os.path.abspath(config_path))
    images_dir = resolve_path(config_dir, ds["images_dir"])

    data_file = resolve_path(config_dir, ds.get("data_file", ""))
    if data_file and os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("data_file 必须是 JSON list")
        samples: List[Sample] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            img_path = item.get("img_path") or item.get("image") or item.get("img")
            if not img_path:
                continue
            img_path = resolve_path(images_dir, img_path)
            pose = item.get("robot_pose")
            if pose is not None and (not isinstance(pose, list) or len(pose) < 6):
                raise ValueError(f"robot_pose 格式错误（需要list且长度>=6）：{item}")
            samples.append(Sample(img_path=img_path, robot_pose=pose))
        return samples

    # fallback：图片按排序 + pos.txt 按行号对应
    img_files = list_images(images_dir, ds["image_extensions"])
    if not img_files:
        raise FileNotFoundError(f"未找到图片：{images_dir}")
    pose_file = resolve_path(config_dir, ds["pose_file"])
    poses = read_pose_txt(pose_file)
    if len(poses) != len(img_files):
        raise ValueError(
            f"图片数量({len(img_files)})与位姿数量({len(poses)})不一致。"
            f"建议使用 dataset.data_file 显式绑定 img_path 与 robot_pose。"
        )
    return [Sample(img_path=p, robot_pose=poses[i]) for i, p in enumerate(img_files)]


# ----------------------------
# 标定板检测与PnP
# ----------------------------

def build_object_points(calib_cfg: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[int, int]]:
    rows = int(calib_cfg["rows"])
    cols = int(calib_cfg["cols"])
    sp = float(calib_cfg["spacing"])
    calib_type = str(calib_cfg.get("type", "chessboard")).lower()

    if calib_type == "chessboard":
        # 内角点数：(rows-1) x (cols-1)，patternSize 传 (cols-1, rows-1)
        pattern_size = (cols - 1, rows - 1)
        objp = np.zeros(((rows - 1) * (cols - 1), 3), np.float32)
        objp[:, :2] = (np.mgrid[0 : cols - 1, 0 : rows - 1].T.reshape(-1, 2) * sp)
        return objp, pattern_size

    # circles：点数 rows x cols，patternSize 传 (cols, rows)
    pattern_size = (cols, rows)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sp)
    return objp, pattern_size


def detect_board_corners(
    img_bgr: np.ndarray,
    calib_cfg: Dict[str, Any],
    pattern_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    calib_type = str(calib_cfg.get("type", "chessboard")).lower()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if calib_type == "chessboard":
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not ret:
            return None
        corners = corners.astype(np.float32)
        if calib_cfg.get("refine_corners", True):
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        return corners

    # circles grid（默认对称圆点阵）
    flags = cv2.CALIB_CB_SYMMETRIC_GRID
    ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags)
    if not ret:
        return None
    return corners.astype(np.float32)


def solve_pnp(
    objp: np.ndarray,
    corners: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    succ, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not succ:
        raise RuntimeError("solvePnP 失败")
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))
    Rm, _ = cv2.Rodrigues(rvec)
    return Rm, tvec, err


def handeye_method_from_cfg(name: str) -> Optional[int]:
    n = (name or "").strip().upper()
    if not n:
        return None
    mapping = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if n not in mapping:
        raise ValueError(f"未知 handeye_method：{name}，可选 {list(mapping.keys())}")
    return mapping[n]


# ----------------------------
# 主流程（保留原算法调用与验证链路）
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hand_eye_config.json", help="配置文件路径(JSON)")
    args = parser.parse_args()

    # 设置 NumPy 的打印选项，禁用科学计数法
    np.set_printoptions(suppress=True)

    cfg = load_config(args.config)
    cam = cfg["camera"]

    fx = float(cam.get("fx", 0))
    fy = float(cam.get("fy", 0))
    cx = float(cam.get("cx", 0))
    cy = float(cam.get("cy", 0))
    if fx == 0 or fy == 0 or cx == 0 or cy == 0:
        raise ValueError("相机内参未填写（fx/fy/cx/cy 不能为0）")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    dist_list = cam.get("dist", [0, 0, 0, 0, 0])
    if len(dist_list) < 5:
        dist_list = list(dist_list) + [0] * (5 - len(dist_list))
    dist = np.array(dist_list[:5], dtype=float).reshape(-1, 1)

    samples = load_samples(cfg, args.config)
    if len(samples) < 3:
        raise ValueError("有效样本不足 3 组，无法进行手眼标定")

    calib_cfg = cfg["calibration"]
    objp, pattern_size = build_object_points(calib_cfg)

    ds = cfg["dataset"]
    angle_unit = str(ds.get("pose_angle_unit", "rad")).lower()
    pose_is_base_to_gripper = bool(ds.get("pose_is_base_to_gripper", True))

    R_gripper2base: List[np.ndarray] = []
    t_gripper2base: List[np.ndarray] = []
    R_target2cam: List[np.ndarray] = []
    t_target2cam: List[np.ndarray] = []
    per_view_reproj: List[float] = []
    used: List[Sample] = []

    for s in samples:
        if not s.robot_pose:
            continue
        img = cv2.imread(s.img_path)
        if img is None:
            continue
        corners = detect_board_corners(img, calib_cfg, pattern_size)
        if corners is None:
            continue

        # board(target) -> camera
        R_tc, t_tc, reproj = solve_pnp(objp, corners, K, dist)
        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)
        per_view_reproj.append(reproj)

        # 机器人位姿 -> gripper->base（OpenCV calibrateHandEye 要求）
        T_input = pose_to_T(s.robot_pose, angle_unit=angle_unit)
        T_gripper_to_base = invert_T(T_input) if pose_is_base_to_gripper else T_input
        R_gripper2base.append(T_gripper_to_base[:3, :3])
        t_gripper2base.append(T_gripper_to_base[:3, 3:4])

        used.append(s)

    if len(used) < 3:
        raise ValueError("可用样本不足 3 组（图片读取/标定板识别/位姿缺失导致）")

    method = handeye_method_from_cfg(cfg.get("handeye_method", ""))
    if method is None:
        # 保留你原脚本的做法：不指定 method，使用 OpenCV 默认
        R, T = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
    else:
        R, T = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method
        )

    print("手眼矩阵分解得到的旋转矩阵")
    print(R)
    print("\n")

    print("手眼矩阵分解得到的平移矩阵")
    print(T)

    RT = np.column_stack((R, T))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))  # 即为 cam to end
    print("\n")
    print("相机相对于末端的变换矩阵为：")
    print(RT)

    out = cfg["output"]
    cam2end_path = resolve_path(os.getcwd(), out.get("cam2end_path", "cam2end.txt"))
    with open(cam2end_path, "w", encoding="utf-8") as f:
        for row in RT:
            f.write(" ".join(map(str, row.tolist())) + "\n")

    # 结果验证：固定标定板在基座下的位姿应当稳定（每次结果相差较小）
    board_in_base_list: List[np.ndarray] = []
    for i in range(len(used)):
        RT_end_to_base = np.row_stack(
            (np.column_stack((R_gripper2base[i], t_gripper2base[i])), np.array([0, 0, 0, 1]))
        )
        RT_chess_to_cam = np.row_stack(
            (np.column_stack((R_target2cam[i], t_target2cam[i])), np.array([0, 0, 0, 1]))
        )
        RT_cam_to_end = RT  # cam->end(gripper)

        # target->base（再求逆得到 base->target）
        RT_chess_to_base = RT_end_to_base @ RT_cam_to_end @ RT_chess_to_cam
        RT_base_to_chess = np.linalg.inv(RT_chess_to_base)
        board_in_base_list.append(RT_base_to_chess)

        print("第", i, "次")
        print(f"{RT_base_to_chess[:3, :]}")
        print("")

    # 输出统计与报告（便于判断“稳定性/准确性”）
    trans = np.array([T[:3, 3] for T in board_in_base_list], dtype=float)
    t_mean = trans.mean(axis=0).tolist()
    t_std = trans.std(axis=0).tolist()
    reproj_mean = float(np.mean(per_view_reproj)) if per_view_reproj else float("nan")

    report = {
        "used_samples": len(used),
        "reprojection_error_mean_px": reproj_mean,
        "board_pose_in_base_translation_mean": t_mean,
        "board_pose_in_base_translation_std": t_std,
        "cam_to_gripper_T": RT.tolist(),
    }
    report_path = resolve_path(os.getcwd(), out.get("report_path", "handeye_report.json"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"[完成] cam2end 已保存：{cam2end_path}")
    print(f"[完成] 报告已保存：{report_path}")
    print(f"[提示] 平均重投影误差：{reproj_mean:.4f} px（越小越好）")


if __name__ == "__main__":
    main()
