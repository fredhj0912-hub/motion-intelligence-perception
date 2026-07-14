import numpy as np
import open3d as o3d
import os
import json
import matplotlib.pyplot as plt

def load_kitti_bin(bin_path):
    """
    KITTI .bin 바이너리 파일을 로드하여 NumPy 포인트 배열(N, 3)과 강도(N, 1)를 반환합니다.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")
    scan = np.fromfile(bin_path, dtype=np.float32)
    point_cloud = scan.reshape((-1, 4))
    points = point_cloud[:, :3]
    intensities = point_cloud[:, 3]
    return points, intensities

def run_perception_pipeline(bin_path, distance_threshold=0.2, eps=0.6, min_points=10):
    """
    지면 분할(RANSAC) + 장애물 클러스터링(DBSCAN) 파이프라인을 실행합니다.
    
    Parameters:
    - bin_path: .bin 파일 경로
    - distance_threshold: RANSAC 지면 분할 거리 임계값 (m)
    - eps: DBSCAN 클러스터링 반경 오차 (m)
    - min_points: DBSCAN 클러스터를 형성하기 위한 최소 포인트 수
    
    Returns:
    - ground_pcd: 지면 포인트클라우드
    - obstacle_pcd: 장애물 포인트클라우드 (클러스터 색상 적용됨)
    - bboxes: 검출된 장애물의 3D 바운딩 박스 객체 목록
    - obstacle_list: 다음 단계를 위한 장애물 정보 딕셔너리 리스트
    """
    # 1. 데이터 로드 및 Open3D 객체 생성
    points, intensities = load_kitti_bin(bin_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # [전처리] 관심 영역(ROI) 필터링 (선택 사항)
    # 너무 멀거나 뒤쪽에 있는 포인트는 예측에 불필요하므로 잘라낼 수 있습니다.
    # 여기서는 x: 0 ~ 45m, y: -15 ~ 15m, z: -2m ~ 3m 범위로 한정합니다.
    min_bound = np.array([0.0, -15.0, -2.5])
    max_bound = np.array([45.0, 15.0, 3.0])
    bbox_roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd = pcd.crop(bbox_roi)
    
    print(f"Original points in ROI: {len(pcd.points)}")
    
    # ---------------------------------------------------------
    # 2. RANSAC 지면 분할 (Ground Segmentation)
    # ---------------------------------------------------------
    # plane_model: 평면 방정식 [A, B, C, D] (Ax + By + Cz + D = 0)
    # inliers: 평면에 속하는 포인트의 인덱스 리스트
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=150
    )
    
    a, b, c, d = plane_model
    print(f"Estimated Ground Plane Equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    
    # 지면과 비지면 포인트 분리
    ground_pcd = pcd.select_by_index(inliers)
    obstacle_pcd = pcd.select_by_index(inliers, invert=True)
    
    # 시각화용 색상 지정 (지면은 차분한 회색)
    ground_pcd.paint_uniform_color([0.3, 0.3, 0.3])
    
    print(f"  - Ground points: {len(ground_pcd.points)}")
    print(f"  - Obstacle candidate points: {len(obstacle_pcd.points)}")
    
    # ---------------------------------------------------------
    # 3. 유클리디안 클러스터링 (DBSCAN)
    # ---------------------------------------------------------
    # labels: 각 포인트의 클러스터 ID 배열 (노이즈는 -1)
    labels = np.array(obstacle_pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=False
    ))
    
    max_label = labels.max()
    num_clusters = max_label + 1
    print(f"Detected {num_clusters} obstacle clusters (excluding noise).")
    
    # 클러스터별로 다른 색상을 매핑하기 위한 칼라맵 생성
    cmap = plt.get_cmap("tab20")
    
    # 장애물 포인트클라우드 색상 배열 초기화 (기본 블랙)
    obstacle_colors = np.zeros((len(obstacle_pcd.points), 3))
    
    bboxes = []
    obstacle_list = []
    
    # ---------------------------------------------------------
    # 4. 각 클러스터 분석 및 바운딩 박스 생성
    # ---------------------------------------------------------
    for i in range(num_clusters):
        # 현재 클러스터에 해당하는 포인트들의 인덱스 추출
        cluster_indices = np.where(labels == i)[0]
        
        # 색상 칠하기 (인덱스 순환)
        color = cmap(i % 20)[:3]
        obstacle_colors[cluster_indices] = color
        
        # 현재 장애물의 포인트클라우드만 따로 추출
        cluster_pcd = obstacle_pcd.select_by_index(cluster_indices)
        
        # 3D Oriented Bounding Box (OBB) 계산
        # OBB는 물체의 회전 방향에 정렬된 최소 크기의 상자입니다.
        try:
            obb = cluster_pcd.get_oriented_bounding_box()
            obb.color = color  # 박스 테두리 색상을 클러스터 색상과 일치시킴
            bboxes.append(obb)
            
            # 다음 주차(궤적 예측)에 입력으로 제공할 중심점, 크기, 회전정보 추출
            center = obb.center.tolist()      # [x, y, z]
            extent = obb.extent.tolist()      # [dx, dy, dz] (상자의 가로, 세로, 높이 크기)
            
            # 회전 행렬 R에서 Yaw (Z축 기준 회전각) 추출
            # R은 3x3 회전행렬이며, Z축 회전(Yaw)은 R[0,0]과 R[1,0]을 통해 구할 수 있습니다.
            R = obb.R
            yaw = np.arctan2(R[1, 0], R[0, 0])
            
            obstacle_list.append({
                "id": int(i),
                "position": center,  # [x, y, z]
                "size": extent,      # [dx, dy, dz]
                "yaw": float(yaw),   # radians
                "point_count": len(cluster_indices)
            })
        except Exception as e:
            # 포인트 개수가 너무 적거나 한 평면에만 몰려있는 경우 OBB 계산이 실패할 수 있습니다.
            print(f"Warning: Failed to compute OBB for cluster {i}: {e}")
            
    # 노이즈 포인트(label == -1)는 흐릿한 어두운 빨간색으로 매핑
    noise_indices = np.where(labels == -1)[0]
    obstacle_colors[noise_indices] = [0.15, 0.05, 0.05]
    
    # 색상 적용
    obstacle_pcd.colors = o3d.utility.Vector3dVector(obstacle_colors)
    
    return ground_pcd, obstacle_pcd, bboxes, obstacle_list

def save_obstacles_to_json(obstacle_list, output_path="detected_obstacles.json"):
    """
    검출된 장애물 정보를 다음 단계를 위해 JSON 파일로 저장합니다.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obstacle_list, f, indent=4, ensure_ascii=False)
    print(f"Saved obstacle data ({len(obstacle_list)} items) to {os.path.abspath(output_path)}")
def main():
    bin_file = "synthetic_scene.bin"
    if not os.path.exists(bin_file) and os.path.exists("../stage_1_raw_visualization/synthetic_scene.bin"):
        bin_file = "../stage_1_raw_visualization/synthetic_scene.bin"
    output_json = "detected_obstacles.json"
    
    # 파이프라인 매개변수 설정
    # RANSAC distance_threshold: 0.15m (15cm 이내의 점들만 지면으로 판단)
    # DBSCAN eps: 0.8m (80cm 이내의 점들은 하나의 장애물로 연결)
    # DBSCAN min_points: 8개 (최소 8개 이상의 점이 모여야 장애물로 인정)
    distance_threshold = 0.15
    eps = 0.8
    min_points = 8
    
    try:
        # 1. 인지 파이프라인 실행
        ground, obstacles, bboxes, obstacle_info = run_perception_pipeline(
            bin_file,
            distance_threshold=distance_threshold,
            eps=eps,
            min_points=min_points
        )
        
        # 2. 장애물 바운딩박스 정보 JSON 저장
        save_obstacles_to_json(obstacle_info, output_json)
        
        # 3. 3D 시각화
        print("\n=== Open3D 3D Viewer (2단계 결과) ===")
        print("색상 안내:")
        print("  - 회색 (Gray) : 검출되어 분리된 도로 지면 (Ground)")
        print("  - 유색 (Colored) : 개별 장애물 클러스터 (Cars / Pedestrians)")
        print("  - 박스 (Boxes) : 각 장애물의 Oriented Bounding Box (OBB)")
        print("  - 어두운 빨간색 (Dark Red) : 노이즈 포인트 (Noise)")
        print("======================================")
        
        # 시각화할 개체들을 리스트로 묶어 전달
        # 지면 + 장애물 포인트들 + 바운딩 박스들
        geometries = [ground, obstacles] + bboxes
        
        # 센서 원점 표시를 위한 3D 좌표축 추가
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        geometries.append(axes)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Motion Intelligence - Perception Pipeline Results",
            width=1280,
            height=800,
            left=50,
            top=50
        )
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
