import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

def load_kitti_bin(bin_path):
    """
    KITTI .bin 바이너리 파일을 로드하여 Open3D PointCloud 객체로 변환합니다.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")
        
    print(f"Loading point cloud from: {bin_path}")
    
    # 1. 파일 읽기: float32 형식의 바이너리 데이터 로드
    # KITTI 벨로다인 포인트클라우드는 각 포인트마다 [x, y, z, intensity] 순으로 기록되어 있습니다.
    scan = np.fromfile(bin_path, dtype=np.float32)
    
    # 2. 형상 변경: (N, 4) 형태로 차원 조절
    point_cloud = scan.reshape((-1, 4))
    
    # 3. 좌표(x, y, z)와 강도(intensity) 분리
    points = point_cloud[:, :3]      # x, y, z
    intensities = point_cloud[:, 3]  # 반사 강도 (Intensity)
    
    # 4. Open3D PointCloud 객체 생성 및 포인트 할당
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 5. [옵션] 시각적 퀄리티를 위해 Intensity 값을 칼라맵(Plasma)으로 변환하여 매핑
    # Intensity 값 범위 정규화 (0.0 ~ 1.0)
    norm_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
    
    # Matplotlib 칼라맵 적용 (Plasma 칼라맵 사용)
    cmap = plt.get_cmap('plasma')
    colors = cmap(norm_intensities)[:, :3]  # RGBA에서 RGB만 선택
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Successfully loaded {len(points)} points.")
    return pcd

def main():
    # 생성된 가상 데이터 경로 설정
    bin_file = "synthetic_scene.bin"
    
    try:
        # 포인트클라우드 로드
        pcd = load_kitti_bin(bin_file)
        
        # 원점에 3D 좌표축(RGB 축: Red=X, Green=Y, Blue=Z) 추가
        # size=2.0m 크기의 좌표축 생성
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        
        # 3D 시각화 실행
        print("\n=== Open3D 3D Viewer ===")
        print("마우스 조작:")
        print("  - 좌클릭 드래그: 회전 (Rotate)")
        print("  - 우클릭 드래그: 평행 이동 (Pan)")
        print("  - 휠 스크롤: 줌 인/아웃 (Zoom)")
        print("키보드 조작:")
        print("  - [ / ] : 포인트 크기 줄이기 / 키우기")
        print("  - R : 카메라 뷰 초기화")
        print("  - H : 도움말 출력")
        print("  - Q 또는 ESC : 창 닫기")
        print("=========================")
        
        o3d.visualization.draw_geometries(
            [pcd, axes],
            window_name="Motion Intelligence - Raw Point Cloud Viewer",
            width=1024,
            height=768,
            left=50,
            top=50
        )
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
