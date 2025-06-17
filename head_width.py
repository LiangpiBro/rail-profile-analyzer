"""
head_width 模块

本模块用于在一个闭合或开放的二维轮廓（contour，类型为 shapely.geometry.LineString）中，
通过分析不同区域（大圆、小圆）与轮廓的交点，测量与中心点相关的几何距离，并确定符合特定几何结构特征的横截面位置（X坐标）。

主要功能包括：
- 在 X 方向扫描轮廓，查找某一横截面，使得从中心点出发与两个特定圆的交点之间的距离接近目标值。
- 通过二分法精确查找目标位置。
- 计算从中心点出发的主向量方向、平行偏移线，并返回其与轮廓的交点，用于评估头部偏移（head offset）是否符合正向要求。

核心函数：
- `head_width(...)`：主入口函数。返回关键几何点、距离、偏移量等信息。
- `measure_dist_at_x(...)`：在指定 X 坐标测量中心点到圆弧交点的距离。
- `binary_search_for_x(...)`：通过二分法精确查找满足目标距离的 X 坐标。
- `compute_parallel_intersection(...)`：计算偏移线与轮廓的交点。
- `segment_circle_intersection(...)`：求线段与圆的交点，支持快速和符号两种方法。
- `vertical_intersections_at_x(...)`：构造垂直线并求与轮廓交点，用于确定扫描中心点。

默认参数说明（head_width）：
- R_big: 大圆半径，默认 50.7
- R_small: 小圆半径，默认 15.81387037
- target_dist: 目标距离（两圆交点之间距离），默认 62.16903165
- positive_head_width: 用于偏移比较的头部标准宽度，默认 71.29999994
- step_length: 扫描步长，默认 0.5
- epsilon: 二分收敛精度，默认 1e-4

依赖库：
- shapely
- numpy
- sympy（仅用于符号解法）

示例用途：
适用于复杂轮廓线段中自动查找某一特定“宽度特征”位置，并用于对比设计值与实际轮廓之间的偏移量。

作者：佟垚
日期：2025年6月11日
"""


from shapely.geometry import LineString, Point
from sympy import symbols, Eq, solve
import numpy as np

import time

# ✅ 3. 主函数（只返回跳变前后两个点）
def head_width(
    contour: LineString,
    R_big=50.7,
    R_small=15.81387037,
    target_dist=62.16903165,
    positive_head_width=71.29999994,
    step_length=0.5,
    epsilon=1e-4
):
    t0 = time.perf_counter()  # ⏱ 记录起始时间
    coords = list(contour.coords)

    def in_range(p, region):
        x, y = p
        if region == "big":
            return 22 <= x <= 28 and 165 <= y <= 185
        elif region == "small":
            return -40 <= x <= -28 and 157 <= y <= 167
        return False

    big_segments = []
    small_segments = []
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        if in_range(p1, "big") or in_range(p2, "big"):
            big_segments.append((p1, p2))
        if in_range(p1, "small") or in_range(p2, "small"):
            small_segments.append((p1, p2))
    t1 = time.perf_counter()
    print(f"大小区域划分完成：big_segments={len(big_segments)}，small_segments={len(small_segments)}")
    print(f"执行用时：{t1 - t0:.6f} 秒")
    prev_result = None
    prev_delta = None

    for x in np.arange(-28, -22.01, step_length):
        result = measure_dist_at_x(contour, x, R_big, R_small, big_segments, small_segments)
        if result is None:
            continue
        _, center, big_point, small_point, dist = result
        delta = dist - target_dist

        if prev_delta is None:
            prev_result = result
            prev_delta = delta
            continue

        if delta * prev_delta < 0:
            x1 = prev_result[0]
            x2 = result[0]
            x_final, center, big_point, small_point, dist = binary_search_for_x(
                contour, x1, x2, R_big, R_small, big_segments, small_segments,
                target_dist=target_dist,
                epsilon=epsilon
            )
            main_line, parallel_line, intersection = compute_parallel_intersection(center, big_point, small_point, contour)

            offset_dist = small_point.distance(intersection) if isinstance(intersection, Point) else None
            offset_diff = positive_head_width - offset_dist   if offset_dist is not None else None
            # print(offset_dist)
            # print(positive_head_width)
            # print(offset_diff)


            return (
                x_final, center, big_point, small_point, dist,
                main_line, parallel_line, intersection,
                offset_dist, offset_diff
            )

        if abs(delta) < abs(prev_delta):
            prev_result = result
            prev_delta = delta
    t2 = time.perf_counter()
    print("找到跳变的点")
    print(f"执行用时：{t2 - t1:.6f} 秒")
    if prev_result:
        x, center, big_point, small_point, dist = prev_result
        main_line, parallel_line, intersection = compute_parallel_intersection(center, big_point, small_point, contour)
        offset_dist = small_point.distance(intersection) if isinstance(intersection, Point) else None
        offset_diff = positive_head_width - offset_dist if offset_dist is not None else None

        return (
            x, center, big_point, small_point, dist,
            main_line, parallel_line, intersection,
            offset_dist, offset_diff
        )

    return None

def compute_parallel_intersection(center, big_point, small_point, contour, extend_length=80):
    vec = np.array([big_point.x - center.x, big_point.y - center.y])
    norm_vec = vec / np.linalg.norm(vec)
    offset_start = np.array([small_point.x, small_point.y]) + norm_vec * (extend_length -30)
    offset_end   = np.array([small_point.x, small_point.y]) + norm_vec * extend_length
    main_line = LineString([center, big_point])
    parallel_line = LineString([offset_start, offset_end])
    intersection = parallel_line.intersection(contour)
    return main_line, parallel_line, intersection


#返回一个X坐标直线和轮廓线的交点
def vertical_intersections_at_x(contour: LineString, x: float, y_margin: float = 10) -> list[Point]:
    """
    在指定 x 值处，构造竖直线并与轮廓 contour 相交，返回所有交点。

    参数：
        contour: LineString - 闭合或开放轮廓线
        x: float - 竖直线的 X 坐标
        y_margin: float - 向上下延伸的范围（默认 10）

    返回：
        List[Point] - 所有交点（可能为空）
    """
    _, _, _, maxy = contour.bounds
    vertical_line = LineString([(x, maxy - y_margin), (x, maxy + y_margin)])
    intersection = contour.intersection(vertical_line)

    if intersection.is_empty:
        return []
    elif intersection.geom_type == 'Point':
        return [intersection]
    elif intersection.geom_type == 'MultiPoint':
        return list(intersection.geoms)
    else:
        return []  # 忽略非点类型（如线段重合等特殊情况）
#返回圆和线段的交点
def segment_circle_intersection(p1, p2, center, radius, method='fast'):
    if method == 'fast':
        # ✅ 数值几何解法（推荐，快）
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = center

        dx = x2 - x1
        dy = y2 - y1
        fx = x1 - cx
        fy = y1 - cy

        a = dx*dx + dy*dy
        b = 2 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - radius*radius

        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None

        discriminant = discriminant**0.5
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        for t in [t1, t2]:
            if 0 <= t <= 1:
                px = x1 + t * dx
                py = y1 + t * dy
                return Point(px, py)

        return None

    elif method == 'slow':
        # 🐢 原来的符号解法（慢，但理论精确）
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = center
        t = symbols('t')
        xt = x1 + t * (x2 - x1)
        yt = y1 + t * (y2 - y1)
        eq = Eq((xt - cx) ** 2 + (yt - cy) ** 2, radius ** 2)
        sols = solve(eq, t)

        for sol in sols:
            if sol.is_real and 0 <= sol <= 1:
                px = float(xt.subs(t, sol).evalf())
                py = float(yt.subs(t, sol).evalf())
                return Point(px, py)

        return None

    else:
        raise ValueError("Invalid method. Use 'fast' or 'slow'.")

# ✅ 1. 单点测量函数
def measure_dist_at_x(contour, x, R_big, R_small, big_segments, small_segments):
    vertical_pts = vertical_intersections_at_x(contour, x)
    if not vertical_pts:
        return None
    center = vertical_pts[0]
    cx, cy = center.x, center.y

    big_point = None
    small_point = None

    for p1, p2 in big_segments:
        d1 = Point(p1).distance(center)
        d2 = Point(p2).distance(center)
        if (d1 < R_big and d2 > R_big) or (d1 > R_big and d2 < R_big):
            big_point = segment_circle_intersection(p1, p2, (cx, cy), R_big, method='fast')
            if big_point:
                break

    for p1, p2 in small_segments:
        d1 = Point(p1).distance(center)
        d2 = Point(p2).distance(center)
        if (d1 < R_small and d2 > R_small) or (d1 > R_small and d2 < R_small):
            small_point = segment_circle_intersection(p1, p2, (cx, cy), R_small,method='slow')
            if small_point:
                break

    if big_point and small_point:
        dist = big_point.distance(small_point)
        return (x, center, big_point, small_point, dist)
    else:
        return None


# ✅ 2. 二分搜索函数
def binary_search_for_x(contour, x1, x2, R_big, R_small, big_segments, small_segments, target_dist=62, epsilon=1e-4, max_iter=50):
    iter_count = 0  # 记录迭代次数
    for _ in range(max_iter):
        iter_count += 1
        x_mid = (x1 + x2) / 2
        result = measure_dist_at_x(contour, x_mid, R_big, R_small, big_segments, small_segments)
        if result is None:
            break
        _, _, _, _, dist = result
        delta = dist - target_dist

        result1 = measure_dist_at_x(contour, x1, R_big, R_small, big_segments, small_segments)
        if result1 is None:
            break
        _, _, _, _, dist1 = result1
        delta1 = dist1 - target_dist

        if abs(delta) < epsilon:
            print(f"✅ 收敛于第 {iter_count} 次迭代")
            return result
        elif delta * delta1 < 0:
            x2 = x_mid
        else:
            x1 = x_mid

    return result  # 返回最后一次的结果（可能是近似）
