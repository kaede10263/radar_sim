import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point

# ==== 牆面開關 ====
enable_top_wall = True
enable_bottom_wall = True
enable_left_wall = True
enable_right_wall = True

# ==== 雷達參數 ====
radar = np.array([0.0, 0.0])
r_max = 200
single_deg = 21  # 改成其他角度也可以

# ==== 目標物 ====
targets = []

circle_center = np.array([30, 5])
circle_radius = 2.0
circle_poly = Point(circle_center).buffer(circle_radius, resolution=32)
targets.append(circle_poly)

rect = Polygon([(20, -10), (26, -10), (26, -5), (20, -5)])
targets.append(rect)

triangle = Polygon([(40, 15), (43, 20), (37, 20)])
targets.append(triangle)

# ==== 牆定義 ====
walls = []
x_wall = 100
y_wall = 30

if enable_bottom_wall:
    walls.append(LineString([(-x_wall, -y_wall), (x_wall, -y_wall)]))
if enable_top_wall:
    walls.append(LineString([(-x_wall, y_wall), (x_wall, y_wall)]))
if enable_left_wall:
    walls.append(LineString([(-x_wall, -y_wall), (-x_wall, y_wall)]))
if enable_right_wall:
    walls.append(LineString([(x_wall, -y_wall), (x_wall, y_wall)]))

# ==== 畫圖初始化 ====
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
plt.grid(True)
plt.title("Radar Reflection Simulation with Walls")
plt.xlabel("X")
plt.ylabel("Y")

for poly in targets:
    x, y = poly.exterior.xy
    ax.plot(x, y, 'k-')

for wall in walls:
    x, y = wall.xy
    ax.plot(x, y, 'b--', linewidth=1, label='Wall')

ax.plot(radar[0], radar[1], 'bo', label='Radar')

# ==== 發射光線 ====
theta = np.deg2rad(single_deg)
dir_vec = np.array([np.cos(theta), np.sin(theta)])
current_point = radar.copy()
current_dir = dir_vec.copy()
total_length = 0
reflection_points = []
reflection_count = 0
first_hit_red_line_drawn = False
theta_list = []  # 儲存紅線回波方向（弧度）

while total_length < r_max:
    ray_end = current_point + current_dir * (r_max - total_length)
    ray = LineString([current_point, ray_end])

    nearest_hit = None
    nearest_dist = np.inf
    nearest_poly = None

    # ==== 搜尋目標 ====
    for poly in targets:
        inter = ray.intersection(poly.boundary)
        if inter.is_empty:
            continue
        if inter.geom_type == 'MultiPoint':
            for p in inter.geoms:
                pt = np.array(p.coords[0])
                dist = np.linalg.norm(pt - current_point)
                if dist < nearest_dist and dist > 1e-6:
                    nearest_hit = pt
                    nearest_dist = dist
                    nearest_poly = poly
        elif inter.geom_type == 'Point':
            pt = np.array(inter.coords[0])
            dist = np.linalg.norm(pt - current_point)
            if dist < nearest_dist and dist > 1e-6:
                nearest_hit = pt
                nearest_dist = dist
                nearest_poly = poly

    # ==== 搜尋牆 ====
    for wall in walls:
        inter = ray.intersection(wall)
        if inter.is_empty:
            continue
        if inter.geom_type == 'Point':
            pt = np.array(inter.coords[0])
            dist = np.linalg.norm(pt - current_point)
            if dist < nearest_dist and dist > 1e-6:
                nearest_hit = pt
                nearest_dist = dist
                nearest_poly = 'wall'

    # ==== 沒碰到任何東西 ====
    if nearest_hit is None:
        final_point = current_point + current_dir * (r_max - total_length)
        ax.plot([current_point[0], final_point[0]], [current_point[1], final_point[1]], 'gray', linewidth=1)
        break

    segment_length = np.linalg.norm(nearest_hit - current_point)
    if total_length + segment_length > r_max:
        final_point = current_point + current_dir * (r_max - total_length)
        ax.plot([current_point[0], final_point[0]], [current_point[1], final_point[1]], 'gray', linewidth=1)
        break

    # ==== 畫灰色線段 ====
    ax.plot([current_point[0], nearest_hit[0]], [current_point[1], nearest_hit[1]], 'gray', linewidth=1)
    total_length += segment_length
    reflection_points.append(nearest_hit.tolist())
    reflection_count += 1

    # ==== 第一個反射點畫紅線 ====
    if not first_hit_red_line_drawn:
        ax.plot([nearest_hit[0], radar[0]], [nearest_hit[1], radar[1]], 'red', linewidth=1.5)
        theta_list.append(np.arctan2(nearest_hit[1] - radar[1], nearest_hit[0] - radar[0]))
        first_hit_red_line_drawn = True

    # ==== 計算反射方向 ====
    if nearest_poly == 'wall':
        if np.abs(nearest_hit[1] - y_wall) < 1e-2 or np.abs(nearest_hit[1] + y_wall) < 1e-2:
            reflect = np.array([current_dir[0], -current_dir[1]])
        else:
            reflect = np.array([-current_dir[0], current_dir[1]])
    else:
        for seg_start, seg_end in zip(list(nearest_poly.exterior.coords[:-1]), list(nearest_poly.exterior.coords[1:])):
            edge = LineString([seg_start, seg_end])
            if edge.distance(Point(nearest_hit)) < 1e-3:
                break
        edge_vec = np.array(seg_end) - np.array(seg_start)
        edge_vec /= np.linalg.norm(edge_vec)
        normal_vec = np.array([-edge_vec[1], edge_vec[0]])
        to_radar = radar - nearest_hit
        if np.dot(normal_vec, to_radar) < 0:
            normal_vec *= -1
        incident = (nearest_hit - current_point)
        incident /= np.linalg.norm(incident)
        reflect = incident - 2 * np.dot(incident, normal_vec) * normal_vec

    # ==== 更新射線位置與方向 ====
    current_point = nearest_hit
    current_dir = reflect

    # ==== 若回到雷達方向，畫紅線並停止 ====
    to_radar = radar - current_point
    to_radar /= np.linalg.norm(to_radar)
    dot_return = np.dot(current_dir, to_radar)
    cos_threshold = np.cos(np.deg2rad(1))

    print(dot_return, cos_threshold)
    if dot_return > cos_threshold:
        print("larger line")
        ax.plot([current_point[0], radar[0]], [current_point[1], radar[1]], 'red', linewidth=1.5)
        theta_list.append(np.arctan2(current_point[1] - radar[1], current_point[0] - radar[0]))
        # break

# ==== 輸出結果 ====
print(f"\n✅ Total reflections: {reflection_count}")
for i, pt in enumerate(reflection_points):
    print(f"  {i+1}. Hit at {pt}")

print("\n📐 Radar return angles (theta in degrees):")
for th in theta_list:
    print(f"  {np.rad2deg(th):.2f}°")

ax.legend()
plt.xlim(-100, 100)
plt.ylim(-40, 40)
plt.show()
