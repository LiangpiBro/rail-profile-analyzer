"""
sort_pcd_points 函数说明
=========================

功能概述：
---------
该函数用于读取一组三维点云（.pcd 格式），对其进行主平面拟合与二维压平，
通过最小外接矩形自动对齐钢轨截面方向，然后结合 CAD 标准图形（DXF 中的点路径）进行轮廓点排序。
最终输出闭合的点云轮廓线与其最小外接矩形，适用于几何比对、模板匹配、轮廓误差分析等任务。

处理流程（与代码编号一致）：
-----------------------------

【第一部分】压平点云，构造局部坐标系
-------------------------------------
① 读取点云数据（Open3D 读取 .pcd 文件）；
② 通过 PCA 提取主平面法向量，对点云进行平面投影（二维压平）；
③ 构建局部坐标系（最大方差方向为 X，垂直方向为 Y）；
④ 利用凸包 + 最小外接矩形识别轨底边；
⑤ 将轨底中点移至原点，并旋转至水平（X 轴方向）；
→ 得到局部对齐后的二维点云坐标 `aligned_points`。

【第二部分】读取标准路径，路径参数化
-------------------------------------
⑥ 从 DXF 文件中提取所有 `POINT` 实体作为标准路径点；
⑦ 基于累计弧长参数化路径（生成 t ∈ [0,1]）；
⑧ 使用线性插值在路径上均匀采样 1000 个点；
→ 得到 `standard_path_points`，作为轮廓排序参考。

【第三部分】路径引导点云排序
----------------------------
⑨ 构建 KDTree，在点云中为每个路径点找到最近邻；
⑩ 对每个点云点，若多个路径点匹配到该点，则按距离远近排序路径点；
→ 构建顺序合理、连续、接近闭合的 `ordered_points`。

【输出】
--------
- `contour_boundary`: 排序后的点云轮廓线（Shapely `LineString`）
- `rectangle_boundary`: 基于排序轮廓的最小外接矩形（Shapely `LineString`）

依赖库：
--------
- open3d：仅用于读取点云
- numpy：核心数据处理
- cv2（OpenCV）：最小外接矩形计算
- ezdxf：读取标准图形 DXF
- scipy：ConvexHull、KDTree、插值
- shapely：构造输出轮廓几何对象
- matplotlib（可选）：调试可视化（已注释）

注意事项：
----------
- DXF 中应包含由 `DIVIDE` 命令生成的等距 `POINT` 点序列；
- 点云应包含较完整、连续的截面轮廓，否则排序可能失效；
- 插值点数（默认 1000）可根据所需精度进行调整；
- 若需要可视化调试，可取消注释相关 `plt` 代码段；
- 仅使用 Open3D 的读取功能，后续处理为纯 2D。

作者建议：
----------
如需扩展为多段轮廓识别、误差匹配、点云质量筛选等模块，可在 `aligned_points` 与 `standard_path_points` 基础上继续开发。
"""

import open3d as o3d
import numpy as np
import cv2
import ezdxf

import matplotlib
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from shapely.geometry import LineString

from collections import defaultdict

matplotlib.use('TkAgg')  # 保留这句
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体（前提是系统已安装）
plt.rcParams['axes.unicode_minus'] = False  # 显示负号时避免乱码


def sort_pcd_points(pcd_file_path):
    #-----------------第一部分-----
    #把点云变成二维
    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    # 获得数据的单位 跨度# points = np.asarray(pcd.points)
    #
    # # ✅ 插入查看坐标范围
    # min_vals = points.min(axis=0)
    # max_vals = points.max(axis=0)
    # ranges = max_vals - min_vals
    #
    # print("坐标范围：")
    # print("X轴：", min_vals[0], "到", max_vals[0], "跨度 =", ranges[0])
    # print("Y轴：", min_vals[1], "到", max_vals[1], "跨度 =", ranges[1])
    # print("Z轴：", min_vals[2], "到", max_vals[2], "跨度 =", ranges[2])

    # 2. 提取点坐标
    points = np.asarray(pcd.points)

    # 3. PCA 拟合平面 找出主方向（法向量）
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2]  # 第三个主方向是法向量（最小方差方向）

    # 4. 将所有点投影到拟合平面上
    projected = points - np.dot((points - center), normal[:, None]) * normal


    #代码第二部分-----把整个钢轨的轨底放在X轴,并把中点放在原点-----------------------------
    #-----------------------------------------------------------------------------

    # 5. 构造局部 2D 坐标系（以拟合平面为基础）

    # 法向量 normal 已有
    # 找到两个在平面上的正交方向作为 2D 坐标轴
    x_axis = vh[0]  # 最大方差方向，作为局部 X
    y_axis = np.cross(normal, x_axis)  # 与法向量和 X 轴正交，作为局部 Y

    # 将每个投影点转换为 2D 坐标（在局部坐标系中）
    # 向 x_axis 和 y_axis 投影
    relative_points = projected - center  # 相对于平面中心
    u_coords = np.dot(relative_points, x_axis)
    v_coords = np.dot(relative_points, y_axis)
    points_2d = np.stack([u_coords, v_coords], axis=1)

    # 使用 ConvexHull 得到边界轮廓
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]

    # 用 OpenCV 计算最小外接矩形（需转成 float32）
    rotated_rect = cv2.minAreaRect(hull_points.astype(np.float32))  # 返回 (中心点, 尺寸, 角度)
    box = cv2.boxPoints(rotated_rect)  # 计算矩形4个顶点
    box = np.array(box)

    # # 可视化：绘点 + 外接矩形
    # plt.figure(figsize=(6, 6))
    # plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, label='Points')
    # plt.plot(*np.append(box, [box[0]], axis=0).T, color='red', linewidth=2, label='Min Area Rect')
    # plt.axis('equal')
    # plt.title("带最小外接矩形的图形")
    # plt.legend()
    # plt.show()


    # 6. 找出 4 条边 + 2 条短边
    edges = []
    for i in range(4):
        pt1 = box[i]
        pt2 = box[(i + 1) % 4]
        length = np.linalg.norm(pt2 - pt1)
        edges.append({'pt1': pt1, 'pt2': pt2, 'length': length})

    edges_sorted = sorted(edges, key=lambda e: e['length'])
    short_edges = edges_sorted[:2]

    # 7. 点到线段距离函数
    def point_to_segment_distance(p, a, b):
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest), closest

    # 8. 计算短边与质心的距离
    centroid_2d = np.mean(points_2d, axis=0)
    distances = []
    for edge in short_edges:
        dist, _ = point_to_segment_distance(centroid_2d, edge['pt1'], edge['pt2'])
        distances.append(dist)

    # 9. 选择距离最远的短边
    chosen_edge = short_edges[np.argmin(distances)]
    pt1 = chosen_edge['pt1']
    pt2 = chosen_edge['pt2']
    chosen_midpoint = (pt1 + pt2) / 2
    chosen_vector = pt2 - pt1

    # 10. 平移 + 旋转
    translated_points = points_2d - chosen_midpoint
    angle = np.arctan2(chosen_vector[1], chosen_vector[0])
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle),  np.cos(-angle)]
    ])
    aligned_points = translated_points @ rotation_matrix.T

    # plt.figure(figsize=(6, 6))
    # plt.scatter(aligned_points[:, 0], aligned_points[:, 1], s=1, label='对齐后的点云')
    # plt.axhline(0, color='gray', linestyle='--')
    # plt.axvline(0, color='gray', linestyle='--')
    # plt.title("对齐：最远短边对齐 X 轴，中点为原点")
    # plt.axis('equal')
    # plt.legend()
    # plt.show()

    #代码第三部分-------------------按照标准廓形把点连接成一个边界,廓形,-----------------------------------
    #-----------------------------------------------------------------------------
    # 11.加载 DXF 文件
    dxf_path = "标准图形.dxf"
    dxf_doc = ezdxf.readfile(dxf_path)
    msp = dxf_doc.modelspace()

    # 提取所有 POINT 实体（DIVIDE 命令产生的点）
    point_list = []

    for entity in msp:
        if entity.dxftype() == "POINT":
            x, y, _ = entity.dxf.location  # 忽略 Z 值
            point_list.append([x, y])

    standard_path_points = np.array(point_list)

    print("提取到的路径点数：", len(standard_path_points))

    if standard_path_points is None:
        raise ValueError("未在 DXF 中找到 LWPOLYLINE 路径，请确认格式正确")

    # 12：路径参数化：累计弧长 → 归一化为 t ∈ [0, 1] -----------------
    # 计算每段路径长度
    diffs = np.diff(standard_path_points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    # 累计长度
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    total_length = cumulative_lengths[-1]
    t_values = cumulative_lengths / total_length  # 参数化进度 t ∈ [0, 1]

    t_interp = np.linspace(0, 1, 1000)  # 均匀采样 1000 个点

    fx = interp1d(t_values, standard_path_points[:, 0], kind='linear')
    fy = interp1d(t_values, standard_path_points[:, 1], kind='linear')

    interp_x = fx(t_interp)
    interp_y = fy(t_interp)
    interp_points = np.stack([interp_x, interp_y], axis=1)

    # ✅ 替换原始变量，统一后续处理（可选）
    standard_path_points = interp_points

    # print("标准路径点数：", len(standard_path_points))
    # print("标准路径总长度：", total_length)
    # print("每段长度（前10个）:", segment_lengths[:10])
    # print("最大段长：", np.max(segment_lengths))
    # print("最小段长：", np.min(segment_lengths))


    # ----------------- ：可视化参数化路径（验证） -----------------
    # plt.figure(figsize=(6, 6))
    # plt.plot(standard_path_points[:, 0], standard_path_points[:, 1], '-', label='插值路径')
    # plt.scatter(standard_path_points[:, 0], standard_path_points[:, 1], c=t_values, cmap='plasma', s=2, label='参数化 t')
    # plt.colorbar(label='t ∈ [0,1]')
    # plt.title('标准路径参数化结果')
    # plt.axis('equal')
    # plt.legend()
    # plt.show()
    #13.匹配----------------------

    # 创建 KDTree 加速最近邻搜索
    tree = cKDTree(aligned_points)

    # 每个路径点匹配到最近的点云点索引
    _, indices = tree.query(standard_path_points, k=1)

    # 构建 点云点索引 → 路径点索引 和距离 的映射
    grouped = defaultdict(list)
    for path_idx, pt_idx in enumerate(indices):
        dist = np.linalg.norm(standard_path_points[path_idx] - aligned_points[pt_idx])
        grouped[pt_idx].append((path_idx, dist))

    # 根据每个点云点下的路径点，按距离排序，再收集路径点索引
    sorted_path_indices = []
    for pt_idx in grouped:
        sorted_entries = sorted(grouped[pt_idx], key=lambda x: x[1])
        for path_idx, _ in sorted_entries:
            sorted_path_indices.append(path_idx)

    # 最终的顺序：路径点顺序（考虑距离微调），对应的点云点集合
    ordered_points = aligned_points[indices[sorted_path_indices]]

    # 首尾闭合处理
    if not np.allclose(ordered_points[0], ordered_points[-1]):
        ordered_points = np.vstack([ordered_points, ordered_points[0]])

    # # 可视化结果：路径引导排序后的轮廓
    # plt.figure(figsize=(6, 6))
    # plt.plot(ordered_points[:, 0], ordered_points[:, 1], '-', label='排序后点云轮廓')
    # plt.scatter(aligned_points[:, 0], aligned_points[:, 1], s=1, alpha=0.3, label='原始点云')
    # plt.axis('equal')
    # plt.title("点云轮廓：路径引导排序结果")
    # plt.legend()
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.scatter(aligned_points[:, 0], aligned_points[:, 1], s=1, label='点云 (aligned_points)', alpha=0.5)
    # plt.plot(standard_path_points[:, 0], standard_path_points[:, 1], '-', color='red', linewidth=1.5, label='参数化路径')
    # plt.legend()
    # plt.axis('equal')
    # plt.title('点云与标准路径的相对位置')
    # plt.grid(True)
    # plt.show()

    # 计算首尾点之间的距离
    # start_point = ordered_points[0]
    # end_point = ordered_points[-1]
    # closure_distance = np.linalg.norm(end_point - start_point)
    #
    # # 设置一个判断闭合的阈值（根据你点云精度选择，一般 1e-2 ~ 1e-1）
    # threshold = 1e-2
    # is_closed = closure_distance < threshold
    #
    # # 输出结果
    # print(f"首尾点距离：{closure_distance:.6f}")
    # print("是否闭合：", "✅ 是闭合轮廓" if is_closed else "❌ 非闭合，建议处理")


    # --- 将 ordered_points 转换为 Shapely Polygon（闭合轮廓） ---

    contour_boundary = LineString(ordered_points)

    # --- 计算最小外接矩形，并转换为 Shapely Polygon ---
    # 注意：这里使用的是基于 ordered_points 的外接矩形，而不是最早那一个
    final_rotated_rect = cv2.minAreaRect(ordered_points.astype(np.float32))
    box = cv2.boxPoints(final_rotated_rect)  # 得到四个角点

    rectangle_boundary = LineString(np.vstack([box, box[0]]))

    # --- 返回两个 Shapely Polygon 对象 ---一个是轮廓，一个是外接矩形
    return contour_boundary, rectangle_boundary