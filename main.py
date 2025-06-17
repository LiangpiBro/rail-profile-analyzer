from process_pcd import sort_pcd_points
from head_width import head_width

import time

import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.use('TkAgg')  # 保留这句
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体（前提是系统已安装）
plt.rcParams['axes.unicode_minus'] = False  # 显示负号时避免乱码

#-------------------代码正文-------------------------
# #-----------保存数据
# pcd_file_path = "2d.pcd"
# contour_boundary, rectangle_boundary = process_pcd(pcd_file_path)
# # 保存轮廓和外接矩形边界
# with open("轮廓数据.pkl", "wb") as f:
#     pickle.dump((contour_boundary, rectangle_boundary), f)

# 程序起始时间
start_time = time.perf_counter()

#-------------打开数据
with open("轮廓数据.pkl", "rb") as f:
    contour_boundary, rectangle_boundary = pickle.load(f)
print("读取完成")
print(f"到达此处总耗时：{time.perf_counter() - start_time:.6f} 秒")


#主代码
result = head_width(contour_boundary)


print("计算完成")
print(f"到达此处总耗时：{time.perf_counter() - start_time:.6f} 秒")
if result is not None:
    (
        x, center, big_point, small_point, dist,
        main_line, parallel_line, intersection,
        offset_dist, offset_diff
    ) = result

    print("将要显示 1 张图")
    print(f"轨头宽度: {offset_dist:.6f} mm，留存缝隙大小为 {offset_diff:.6f} mm")

    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制轮廓
    x_coords, y_coords = contour_boundary.xy
    ax.plot(x_coords, y_coords, color='black', linewidth=1, label='轮廓')

    # 圆
    big_circle = plt.Circle((center.x, center.y), 50.2, color='red', fill=False, linestyle='--', label='大圆 R=50.2')
    small_circle = plt.Circle((center.x, center.y), 16.0216, color='blue', fill=False, linestyle='--', label='小圆 R=16.02')
    ax.add_patch(big_circle)
    ax.add_patch(small_circle)

    # 圆交点（颜色不同）
    ax.plot(big_point.x, big_point.y, 'ro', markersize=6, label='右上方接触点')
    ax.plot(small_point.x, small_point.y, 'bo', markersize=6, label='左下方接触点')

    # 圆心（绿色）
    ax.plot(center.x, center.y, 'go', markersize=6, label='左上方接触点')

    # 主方向线
    ax.plot(*main_line.xy, linestyle='--', color='orange', linewidth=1.2, label='上端两个接触点连线')

    # 平行线交点
    if not intersection.is_empty and intersection.geom_type == "Point":
        ax.plot(intersection.x, intersection.y, 'mo', markersize=6, label='右侧测量点')

        # 连线：small_point → intersection
        ax.plot(
            [small_point.x, intersection.x],
            [small_point.y, intersection.y],
            'g--',
            label='测量基准线'
        )

        # 标注 offset_dist
        text_x = (small_point.x + intersection.x) / 2
        text_y = (small_point.y + intersection.y) / 2
        ax.text(text_x, text_y, f"{offset_dist:.6f} mm", color='green', fontsize=10, ha='center')

    ax.set_title(f"X = {x:.6f} 的分析结果")
    ax.axis('equal')    
    ax.grid(True)
    ax.legend()
    print("加载图片完成")
    print(f"到达此处总耗时：{time.perf_counter() - start_time:.6f} 秒")
    plt.show()

else:
    print("没有找到符合要求的结果。")




