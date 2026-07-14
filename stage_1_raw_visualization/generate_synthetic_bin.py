import numpy as np
import os

def generate_synthetic_lidar_data(output_path="synthetic_scene.bin"):
    """
    가상의 LiDAR 포인트클라우드 데이터를 생성하여 KITTI (.bin) 형식으로 저장합니다.
    KITTI 포맷: 각 포인트는 [x, y, z, intensity]의 4개 float32 값으로 구성됩니다.
    
    LiDAR 센서의 위치는 (0, 0, 0)이고 높이는 지면으로부터 1.73m 위에 있다고 가정합니다.
    즉, 지면(Road)은 z = -1.73 부근에 평면 형태로 생성됩니다.
    """
    print("Generating synthetic LiDAR scene...")
    
    np.random.seed(42)
    points_list = []
    
    # ---------------------------------------------------------
    # 1. 지면 평면 (Ground Plane) 생성
    # ---------------------------------------------------------
    # 센서 기준 전방(x: 2 ~ 40m), 좌우(y: -15 ~ 15m) 영역에 격자 모양으로 배치
    x_grid = np.arange(2.0, 40.0, 0.2)
    y_grid = np.arange(-15.0, 15.0, 0.2)
    
    xx, yy = np.meshgrid(x_grid, y_grid)
    xx = xx.flatten()
    yy = yy.flatten()
    
    # 실제 도로는 미세한 굴곡이나 경사가 있을 수 있으므로 노이즈와 기울기를 살짝 줌
    # 기본 지면 높이 z = -1.73 (KITTI Velodyne 설치 높이 표준)
    zz = -1.73 + 0.005 * xx + np.random.normal(0, 0.02, size=xx.shape)
    
    # 반사도(Intensity): 도로 아스팔트는 반사율이 비교적 낮음 (예: 0.2 부근)
    intensity = np.random.normal(0.2, 0.05, size=xx.shape)
    intensity = np.clip(intensity, 0.0, 1.0)
    
    ground_points = np.stack([xx, yy, zz, intensity], axis=1)
    points_list.append(ground_points)
    
    # ---------------------------------------------------------
    # 2. 장애물 (Obstacles) 생성
    # ---------------------------------------------------------
    # LiDAR는 장애물의 외부 표면(Sensor-facing surface)만 스캔하므로, 
    # 박스의 표면에만 포인트가 생성되도록 모델링합니다.
    
    def generate_box_surface_points(center, size, num_points, intensity_val=0.8):
        """
        주어진 중심(center)과 크기(size)를 가진 박스의 표면에 무작위 포인트를 생성합니다.
        """
        cx, cy, cz = center
        dx, dy, dz = size
        
        # 6개 면에 고르게 포인트 분포
        pts_per_face = num_points // 6
        face_points = []
        
        # 앞/뒤 면 (x = cx +- dx/2)
        for offset in [-dx/2, dx/2]:
            y = np.random.uniform(cy - dy/2, cy + dy/2, pts_per_face)
            z = np.random.uniform(cz - dz/2, cz + dz/2, pts_per_face)
            x = np.full_like(y, cx + offset)
            face_points.append(np.stack([x, y, z], axis=1))
            
        # 좌/우 면 (y = cy +- dy/2)
        for offset in [-dy/2, dy/2]:
            x = np.random.uniform(cx - dx/2, cx + dx/2, pts_per_face)
            z = np.random.uniform(cz - dz/2, cz + dz/2, pts_per_face)
            y = np.full_like(x, cy + offset)
            face_points.append(np.stack([x, y, z], axis=1))
            
        # 상/하 면 (z = cz +- dz/2)
        for offset in [-dz/2, dz/2]:
            x = np.random.uniform(cx - dx/2, cx + dx/2, pts_per_face)
            y = np.random.uniform(cy - dy/2, cy + dy/2, pts_per_face)
            z = np.full_like(x, cz + offset)
            face_points.append(np.stack([x, y, z], axis=1))
            
        box_pts = np.vstack(face_points)
        # 측정 노이즈 추가
        box_pts += np.random.normal(0, 0.015, size=box_pts.shape)
        
        # Intensity 칼럼 추가
        intensities = np.random.normal(intensity_val, 0.05, size=(box_pts.shape[0], 1))
        intensities = np.clip(intensities, 0.0, 1.0)
        
        return np.hstack([box_pts, intensities])
    
    # 장애물 1: 앞쪽에 위치한 승용차 (Car 1)
    # 크기: 길이(dx)=4.0m, 너비(dy)=1.8m, 높이(dz)=1.4m
    # 위치: x=15m (앞으로 15m), y=1.5m (약간 우측), z = 지면 위 (-1.73 + 1.4/2 = -1.03)
    car1 = generate_box_surface_points(
        center=[15.0, 1.5, -1.03],
        size=[4.0, 1.8, 1.4],
        num_points=1200,
        intensity_val=0.7
    )
    points_list.append(car1)
    
    # 장애물 2: 우측 도로변의 SUV 차량 (Car 2)
    # 크기: 길이=4.5m, 너비=2.0m, 높이=1.6m
    # 위치: x=25m, y=5.0m, z = -0.93
    car2 = generate_box_surface_points(
        center=[25.0, 5.0, -0.93],
        size=[4.5, 2.0, 1.6],
        num_points=1000,
        intensity_val=0.65
    )
    points_list.append(car2)
    
    # 장애물 3: 좌측 인도의 보행자 (Pedestrian 1)
    # 크기: dx=0.6m, dy=0.6m, dz=1.7m
    # 위치: x=8.0m, y=-3.5m, z = -0.88
    pedestrian1 = generate_box_surface_points(
        center=[8.0, -3.5, -0.88],
        size=[0.6, 0.6, 1.7],
        num_points=400,
        intensity_val=0.4
    )
    points_list.append(pedestrian1)

    # 장애물 3-2: 승용차 1 바로 옆에 서 있는 보행자 (Pedestrian 2 - 가깝게 배치)
    # 승용차 1이 [15.0, 1.5]에 있고 너비가 1.8m이므로 차량의 왼쪽 끝은 y = 0.6m입니다.
    # 보행자 2를 [15.0, -0.3]에 두고 너비를 0.6m로 설정하여 보행자의 오른쪽 끝은 y = 0.0m가 됩니다.
    # 따라서 차량과 보행자 사이의 순수 공백 거리는 정확히 0.6m입니다.
    pedestrian2 = generate_box_surface_points(
        center=[15.0, -0.3, -0.88],
        size=[0.6, 0.6, 1.7],
        num_points=400,
        intensity_val=0.45
    )
    points_list.append(pedestrian2)
    
    # 장애물 4: 도로 중앙의 큰 장애물 (예: 공사중 드럼통/안전펜스 군집)
    # 크기: dx=2.0m, dy=1.0m, dz=1.2m (주성분 분석 축 안정을 위해 직사각형으로 수정)
    # 위치: x=30m, y=-1.0m, z = -1.13
    construction_obstacle = generate_box_surface_points(
        center=[30.0, -1.0, -1.13],
        size=[2.0, 1.0, 1.2],
        num_points=600,
        intensity_val=0.8
    )
    points_list.append(construction_obstacle)
    
    # ---------------------------------------------------------
    # 3. 데이터 결합 및 저장
    # ---------------------------------------------------------
    all_points = np.vstack(points_list).astype(np.float32)
    
    # KITTI .bin 파일로 출력
    # numpy.ndarray.tofile은 데이터를 바이너리 형식으로 그대로 디스크에 씁니다.
    all_points.tofile(output_path)
    
    print(f"Successfully generated synthetic point cloud data:")
    print(f"  - Output Path: {os.path.abspath(output_path)}")
    print(f"  - Total Points: {all_points.shape[0]}")
    print(f"  - Data Shape: {all_points.shape}")
    print(f"  - Points layout: [x, y, z, intensity]")
    print(f"    - Ground Points: {ground_points.shape[0]}")
    print(f"    - Car 1 Points: {car1.shape[0]}")
    print(f"    - Car 2 Points: {car2.shape[0]}")
    print(f"    - Pedestrian 1 Points: {pedestrian1.shape[0]}")
    print(f"    - Construction Points: {construction_obstacle.shape[0]}")

if __name__ == "__main__":
    generate_synthetic_lidar_data()
