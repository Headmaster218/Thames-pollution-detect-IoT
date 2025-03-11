import geopandas as gpd
import folium
import numpy as np
import matplotlib.pyplot as plt
import base64
from folium.plugins import TimestampedGeoJson
from datetime import datetime, timedelta
from io import BytesIO

# 读取并转换 Shapefile
shapefile_path = "oprvrs_essh_gb/data/WatercourseLink.shp"
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)  # 确保使用 WGS84 坐标系

# 过滤 River Lee 数据
riverline = gdf[gdf["name1"] == "River Thames"]
if riverline.empty:
    raise ValueError("未找到 River Lee 数据")

# 提取河流坐标
def extract_coords(geom):
    if geom.geom_type == "LineString":
        return [[(lat, lon) for lon, lat, *_ in geom.coords]]
    elif geom.geom_type == "MultiLineString":
        return [[(lat, lon) for lon, lat, *_ in line.coords] for line in geom.geoms]
    return []

river_segments = []
for geom in riverline.geometry:
    river_segments.extend(extract_coords(geom))

# 计算河流中心
all_coords = [p for segment in river_segments for p in segment]
center_lat, center_lon = np.mean([p[0] for p in all_coords]), np.mean([p[1] for p in all_coords])

# 创建 Folium 地图
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

# 绘制 River Lea
# 分段绘制 River Thames
for segment in river_segments:
    folium.PolyLine(segment, color="blue", weight=3, opacity=0.8).add_to(m)


# 监测点数据（每个点不同Time的水质数据）
np.random.seed(42)
base_time = datetime(2025, 2, 24, 8, 0)
time_steps = [base_time + timedelta(hours=i) for i in range(6)]

monitoring_points = [
    {"location": (51.55, -0.025), "data": {t: {"pH": np.random.uniform(6.8, 7.5),
                                                  "Turbidity": np.random.uniform(3.0, 5.0),
                                                  "DO2": np.random.uniform(5.5, 7.0),
                                                  "Conducticity": np.random.uniform(240, 280)}
                                              for t in time_steps}},
    {"location": (51.53, -0.018), "data": {t: {"pH": np.random.uniform(6.5, 7.2),
                                                  "Turbidity": np.random.uniform(3.5, 4.5),
                                                  "DO2": np.random.uniform(5.0, 6.5),
                                                  "Conducticity": np.random.uniform(230, 270)}
                                              for t in time_steps}},
    {"location": (51.52, -0.010), "data": {t: {"pH": np.random.uniform(6.9, 7.4),
                                                  "Turbidity": np.random.uniform(3.2, 4.2),
                                                  "DO2": np.random.uniform(5.8, 6.8),
                                                  "Conducticity": np.random.uniform(250, 290)}
                                              for t in time_steps}},
]

# 生成 2×2 的波形图
def generate_plot(data):
    param_list = ["pH", "Turbidity", "DO2", "Conducticity"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))  # 增大图像大小

    for ax, param in zip(axes.flat, param_list):
        ax.plot(time_steps, [data[t][param] for t in time_steps], marker="o", linestyle="-", label=param)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel(param, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid()
        ax.tick_params(axis="x", rotation=30, labelsize=8)  # 旋转 x 轴刻度，缩小字体防止溢出

    plt.tight_layout()  # 防止子图重叠

    # 保存图片到内存
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)  # 增加 dpi 让图更清晰
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # 生成 HTML 图片标签 + CSS 限制大小
    return f"""
    <div style='max-width: 700px; max-height: 600px; overflow: auto;'>
        <img src='data:image/png;base64,{encoded_img}' style='width:100%; height:auto;'>
    </div>
    """


# 转换为 TimestampedGeoJson 格式
features = []
for point in monitoring_points:
    plot_html = generate_plot(point["data"])
    for time, values in point["data"].items():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [point["location"][1], point["location"][0]]
            },
            "properties": {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "popup": Popup(f"{plot_html}<br>"
                            f"Time: {time.strftime('%Y-%m-%d %H:%M')}<br>"
                            f"pH: {values['pH']:.2f}<br>"
                            f"Turbidity: {values['Turbidity']:.2f} NTU<br>"
                            f"DO2: {values['DO2']:.2f} mg/L<br>"
                            f"Conducticity: {values['Conducticity']:.2f} μS/cm", max_width=650),
                "icon": "circle"
            }
        }
        features.append(feature)

geojson_data = {
    "type": "FeatureCollection",
    "features": features
}

# 添加Time选择功能
TimestampedGeoJson(
    geojson_data,
    period="PT1H",  # 每1小时更新
    add_last_point=True,
    auto_play=False,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options="YYYY-MM-DD HH:mm",
    time_slider_drag_update=True
).add_to(m)

# 保存地图
m.save("riverline_time_map.html")
m