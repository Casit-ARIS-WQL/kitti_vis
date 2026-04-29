"""
KITTI 3D目标检测可视化脚本
将3D检测框投影到RGB图像上并可视化。

用法:
    # 可视化所有图像
    python visualize_3d_boxes.py

    # 可视化单张图像
    python visualize_3d_boxes.py --image_id 000145

    # 使用逐帧标定目录（默认即为 calib_gt）
    python visualize_3d_boxes.py --calib_dir calib_gt

    # 使用单个全局标定文件（优先级低于逐帧标定目录）
    python visualize_3d_boxes.py --calib calib/000145.txt

    # 指定输入输出目录
    python visualize_3d_boxes.py --image_dir image --label_dir gt --output_dir output

标注文件格式 (KITTI):
    idx. type truncated occluded alpha left top right bottom height width length x y z rotation_y
"""

import argparse
import os
import numpy as np
import cv2

# 类别颜色映射 (BGR)
CLASS_COLORS = {
    "Car": (0, 255, 0),
    "Van": (0, 165, 255),
    "Truck": (0, 0, 255),
    "Pedestrian": (255, 0, 0),
    "Person_sitting": (255, 128, 0),
    "Cyclist": (255, 0, 255),
    "Tram": (0, 255, 255),
    "Misc": (128, 128, 0),
    "DontCare": (128, 128, 128),
}
DEFAULT_COLOR = (255, 255, 0)

# 默认KITTI P2投影矩阵 (3×4)，来自KITTI训练集典型标定值
# 如无标定文件，将使用此矩阵
DEFAULT_P2 = np.array(
    [
        [721.5377, 0.0, 609.5593, 44.85728],
        [0.0, 721.5377, 172.854, 0.2163791],
        [0.0, 0.0, 1.0, 0.002745884],
    ],
    dtype=np.float64,
)

# 3D框的12条边（顶点索引对）
# 顶点顺序：底面 0-3（前左、前右、后右、后左），顶面 4-7
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7),  # 竖直棱
]


def parse_label_file(label_path):
    """解析KITTI标注文件，返回目标列表。

    每个目标为字典，包含:
        type, truncated, occluded, alpha,
        bbox2d (left, top, right, bottom),
        dimensions (h, w, l),
        location (x, y, z),  # 相机坐标系
        rotation_y
    """
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 支持以 "idx." 开头或直接以类别名开头的格式
            start = 1 if parts[0].endswith(".") else 0
            if len(parts) - start < 15:
                continue
            p = parts[start:]
            obj = {
                "type": p[0],
                "truncated": float(p[1]),
                "occluded": int(p[2]),
                "alpha": float(p[3]),
                "bbox2d": (float(p[4]), float(p[5]), float(p[6]), float(p[7])),
                "dimensions": (float(p[8]), float(p[9]), float(p[10])),  # h, w, l
                "location": (float(p[11]), float(p[12]), float(p[13])),  # x, y, z
                "rotation_y": float(p[14]),
            }
            objects.append(obj)
    return objects


def parse_calib_file(calib_path):
    """解析KITTI标定文件，返回P2矩阵 (3×4)。"""
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_path}")
    with open(calib_path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                values = list(map(float, line.strip().split()[1:]))
                return np.array(values, dtype=np.float64).reshape(3, 4)
    raise ValueError(f"标定文件中未找到 P2 矩阵: {calib_path}")


def get_3d_box_corners(h, w, l, x, y, z, rotation_y):
    """计算3D框的8个顶点在相机坐标系中的位置。

    KITTI坐标系：x向右，y向下，z向前
    location (x, y, z) 是底面中心点（y轴方向朝下，所以底面y=y，顶面y=y-h）

    顶点顺序（从上方看顺时针，底面0-3，顶面4-7）:
        0: 前左底, 1: 前右底, 2: 后右底, 3: 后左底
        4: 前左顶, 5: 前右顶, 6: 后右顶, 7: 后左顶
    """
    # 目标局部坐标系中的8个角点（以底面中心为原点）
    x_c = np.array([l / 2,  l / 2, -l / 2, -l / 2,  l / 2,  l / 2, -l / 2, -l / 2])
    y_c = np.array([0,      0,      0,       0,      -h,     -h,     -h,     -h])
    z_c = np.array([w / 2, -w / 2, -w / 2,  w / 2,  w / 2, -w / 2, -w / 2,  w / 2])

    # 绕Y轴旋转 rotation_y
    cos_ry = np.cos(rotation_y)
    sin_ry = np.sin(rotation_y)
    R = np.array(
        [
            [cos_ry,  0, sin_ry],
            [0,       1, 0],
            [-sin_ry, 0, cos_ry],
        ]
    )

    corners = R @ np.vstack([x_c, y_c, z_c])  # (3, 8)
    corners[0] += x
    corners[1] += y
    corners[2] += z
    return corners  # (3, 8)


def project_to_image(corners_3d, P2):
    """将3D角点 (3×8) 投影到图像平面，返回像素坐标 (8×2)。"""
    ones = np.ones((1, corners_3d.shape[1]))
    corners_hom = np.vstack([corners_3d, ones])  # (4, 8)
    pts_2d = P2 @ corners_hom  # (3, 8)
    pts_2d[:2] /= pts_2d[2]
    return pts_2d[:2].T  # (8, 2)


def draw_3d_box(image, pts_2d, color, thickness=2):
    """在图像上绘制3D框的12条投影边。"""
    pts = pts_2d.astype(int)
    for i, j in BOX_EDGES:
        pt1 = tuple(pts[i])
        pt2 = tuple(pts[j])
        cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)


def draw_2d_box(image, bbox2d, color, thickness=1):
    """在图像上绘制2D检测框。"""
    left, top, right, bottom = (int(v) for v in bbox2d)
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)


def draw_label(image, text, bbox2d, color):
    """在2D框左上角绘制类别标签。"""
    left, top = int(bbox2d[0]), int(bbox2d[1])
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    top_text = max(top - 2, th + 2)
    cv2.rectangle(
        image,
        (left, top_text - th - baseline),
        (left + tw, top_text + baseline),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        text,
        (left, top_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )


def visualize(image_path, label_path, output_path, P2, draw_2d=True):
    """对单张图像进行3D框可视化并保存结果。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [警告] 无法读取图像: {image_path}")
        return

    objects = parse_label_file(label_path)
    if not objects:
        print(f"  [信息] 标注文件为空，跳过: {label_path}")
        cv2.imwrite(output_path, image)
        return

    for obj in objects:
        if obj["type"] == "DontCare":
            continue

        color = CLASS_COLORS.get(obj["type"], DEFAULT_COLOR)
        h, w, l = obj["dimensions"]
        x, y, z = obj["location"]
        ry = obj["rotation_y"]

        # 仅当目标在相机前方时才绘制3D框
        if z > 0:
            corners_3d = get_3d_box_corners(h, w, l, x, y, z, ry)
            pts_2d = project_to_image(corners_3d, P2)
            draw_3d_box(image, pts_2d, color, thickness=2)

        if draw_2d:
            draw_2d_box(image, obj["bbox2d"], color, thickness=1)

        draw_label(image, obj["type"], obj["bbox2d"], color)

    cv2.imwrite(output_path, image)
    print(f"  已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="KITTI 3D目标检测框可视化")
    parser.add_argument(
        "--image_dir",
        default="image",
        help="图像目录（默认: image）",
    )
    parser.add_argument(
        "--label_dir",
        default="gt",
        help="标注文件目录（默认: gt）",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="输出目录（默认: output）",
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="KITTI标定文件路径（可选，包含 P2 矩阵）；"
             "优先级低于 --calib_dir，高于内置默认P2矩阵",
    )
    parser.add_argument(
        "--calib_dir",
        default="calib_gt",
        help="逐帧标定文件目录（默认: calib_gt）；"
             "每帧将自动加载 <calib_dir>/<frame_id>.txt 中的 P2 矩阵",
    )
    parser.add_argument(
        "--image_id",
        default=None,
        help="仅可视化指定帧（如 000145）；不指定则处理所有帧",
    )
    parser.add_argument(
        "--no_2d",
        action="store_true",
        help="不绘制2D检测框（默认同时绘制2D和3D框）",
    )
    args = parser.parse_args()

    # 全局P2矩阵（当逐帧标定文件不可用时的后备）
    global_P2 = None
    if args.calib:
        global_P2 = parse_calib_file(args.calib)
        print(f"使用标定文件中的全局 P2 矩阵: {args.calib}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集待处理的帧列表
    if args.image_id:
        frame_ids = [args.image_id]
    else:
        if not os.path.isdir(args.image_dir):
            parser.error(f"图像目录不存在: {args.image_dir}")
        frame_ids = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(args.image_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        )

    print(f"共找到 {len(frame_ids)} 帧，开始可视化...\n")

    for fid in frame_ids:
        image_path = os.path.join(args.image_dir, fid + ".png")
        if not os.path.exists(image_path):
            image_path = os.path.join(args.image_dir, fid + ".jpg")
        label_path = os.path.join(args.label_dir, fid + ".txt")
        output_path = os.path.join(args.output_dir, fid + ".png")

        if not os.path.exists(image_path):
            print(f"  [警告] 找不到图像文件: {fid}.png / {fid}.jpg，跳过")
            continue
        if not os.path.exists(label_path):
            print(f"  [警告] 找不到标注文件: {label_path}，跳过")
            continue

        # 优先使用逐帧标定文件，其次全局标定文件，最后内置默认值
        frame_calib_path = os.path.join(args.calib_dir, fid + ".txt")
        if os.path.exists(frame_calib_path):
            P2 = parse_calib_file(frame_calib_path)
            calib_source = f"逐帧标定 ({frame_calib_path})"
        elif global_P2 is not None:
            P2 = global_P2
            calib_source = f"全局标定 ({args.calib})"
        else:
            P2 = DEFAULT_P2
            calib_source = "内置默认 P2"

        print(f"处理: {fid}  [{calib_source}]")
        visualize(image_path, label_path, output_path, P2, draw_2d=not args.no_2d)

    print("\n全部完成！")


if __name__ == "__main__":
    main()
