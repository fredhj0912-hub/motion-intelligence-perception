import os
import numpy as np
import open3d as o3d
from flask import Flask, request, jsonify, send_from_directory
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder="static", static_url_path="")

# 가상 데이터 파일 경로
BIN_FILE_PATH = os.path.join("..", "stage_1_raw_visualization", "synthetic_scene.bin")

def load_and_preprocess_pcd(bin_path, voxel_size=0.0):
    """
    KITTI 바이너리 파일을 로드하고 관심 영역(ROI) 필터링 및 복셀 다운샘플링을 적용합니다.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"LiDAR data not found at: {os.path.abspath(bin_path)}")
        
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 관심 영역(ROI) 설정: 전방 0~45m, 좌우 -15~15m, 높이 -2.5~3m
    min_bound = np.array([0.0, -15.0, -2.5])
    max_bound = np.array([45.0, 15.0, 3.0])
    bbox_roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd = pcd.crop(bbox_roi)
    
    # 복셀 다운샘플링 (선택 사항)
    if voxel_size > 0.01:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
    return pcd

@app.route("/")
def index():
    """웹 프론트엔드 메인 페이지를 제공합니다."""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/process", methods=["GET"])
def process_point_cloud():
    """
    파라미터(RANSAC threshold, DBSCAN eps, DBSCAN min_points, voxel_size)를 받아
    지면 분할 및 클러스터링을 실행하고 웹에서 시각화할 수 있는 JSON 형태로 반환합니다.
    """
    # 1. 쿼리 파라미터 로드 및 타입 변환
    try:
        distance_threshold = float(request.args.get("distance_threshold", 0.15))
        eps = float(request.args.get("eps", 0.8))
        min_points = int(request.args.get("min_points", 8))
        voxel_size = float(request.args.get("voxel_size", 0.0))
    except ValueError:
        return jsonify({"error": "Invalid parameter values"}), 400

    try:
        # 2. 포인트클라우드 로드 및 ROI 필터링
        pcd = load_and_preprocess_pcd(BIN_FILE_PATH, voxel_size)
        raw_points = np.asarray(pcd.points).tolist()
        
        # 3. RANSAC 지면 분할
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=150
        )
        
        ground_pcd = pcd.select_by_index(inliers)
        obstacle_pcd = pcd.select_by_index(inliers, invert=True)
        
        ground_points = np.asarray(ground_pcd.points).tolist()
        obstacle_points_array = np.asarray(obstacle_pcd.points)
        
        # 4. DBSCAN 클러스터링
        labels = np.array(obstacle_pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        ))
        
        # 5. 결과 가공 (Three.js 렌더링에 알맞게 포맷팅)
        max_label = int(labels.max()) if len(labels) > 0 else -1
        num_clusters = int(max_label + 1)
        
        obstacles = []
        bboxes = []
        obstacle_list_for_export = []
        
        cmap = plt.get_cmap("tab20")
        
        # 각 클러스터별 포인트 분류 및 OBB 계산
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue
                
            cluster_pts = obstacle_points_array[cluster_indices]
            color = [float(c) for c in cmap(i % 20)[:3]]
            
            # 클러스터 포인트 전송 데이터 구성
            obstacles.append({
                "cluster_id": int(i),
                "color": color,
                "points": cluster_pts.tolist()
            })
            
            # Bounding Box (OBB) 계산
            cluster_pcd = obstacle_pcd.select_by_index(cluster_indices)
            try:
                obb = cluster_pcd.get_oriented_bounding_box()
                center = obb.center.tolist()
                extent = obb.extent.tolist()
                R = obb.R
                yaw = np.arctan2(R[1, 0], R[0, 0])
                
                # 시각화용 Bounding Box 데이터
                bboxes.append({
                    "id": int(i),
                    "center": center,
                    "size": extent,
                    "yaw": float(yaw),
                    "R": obb.R.tolist(),
                    "color": color
                })
                
                # 저장/추출용 데이터
                obstacle_list_for_export.append({
                    "id": int(i),
                    "position": center,
                    "size": extent,
                    "yaw": float(yaw),
                    "point_count": len(cluster_indices)
                })
            except Exception as ex:
                print(f"Failed to generate box for cluster {i}: {ex}")
                
        # 노이즈 포인트(label == -1) 분리 (어두운 주황/빨강으로 렌더링하도록 제공)
        noise_indices = np.where(labels == -1)[0]
        noise_points = obstacle_points_array[noise_indices].tolist() if len(noise_indices) > 0 else []
        
        # 6. JSON 응답 전송
        response_data = {
            "raw_points": raw_points,
            "ground_points": ground_points,
            "obstacles": obstacles,
            "noise_points": noise_points,
            "bboxes": bboxes,
            "export_data": obstacle_list_for_export,
            "stats": {
                "total_points": len(raw_points),
                "ground_points_count": len(ground_points),
                "obstacle_points_count": len(obstacle_points_array) - len(noise_points),
                "noise_points_count": len(noise_points),
                "num_clusters": num_clusters
            }
        }
        
        return jsonify(response_data)
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/export", methods=["POST"])
def export_json():
    """
    프론트엔드에서 현재 필터링된 장애물 JSON 데이터를 받아 로컬 파일로 저장합니다.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400
        
    try:
        # stage_2 폴더에 detected_obstacles.json 갱신 저장
        output_dir = os.path.join("..", "stage_2_segmentation_clustering")
        output_file = os.path.join(output_dir, "detected_obstacles.json")
        
        # 폴더가 없을 경우 생성
        os.makedirs(output_dir, exist_ok=True)
        
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        return jsonify({"message": f"Successfully exported to {os.path.abspath(output_file)}"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

if __name__ == "__main__":
    # 로컬 Flask 개발 서버 가동
    # 로컬호스트 5000 포트에서 동작
    app.run(host="127.0.0.1", port=5000, debug=True)
