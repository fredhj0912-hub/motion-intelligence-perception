// Motion Intelligence - Perception Web Dashboard Logic

// 글로벌 상태 객체
let sceneRaw, cameraRaw, rendererRaw, controlsRaw;
let sceneSeg, cameraSeg, rendererSeg, controlsSeg;
let isSyncing = false; // 카메라 동기화 무한루프 방지 플래그
let exportData = null; // 저장할 장애물 JSON 데이터

// UI 요소
const paramVoxel = document.getElementById("param-voxel");
const paramThreshold = document.getElementById("param-threshold");
const paramEps = document.getElementById("param-eps");
const paramMinpts = document.getElementById("param-minpts");

const valVoxel = document.getElementById("val-voxel");
const valThreshold = document.getElementById("val-threshold");
const valEps = document.getElementById("val-eps");
const valMinpts = document.getElementById("val-minpts");

const statTotal = document.getElementById("stat-total");
const statGround = document.getElementById("stat-ground");
const statObstacles = document.getElementById("stat-obstacles");
const statClusters = document.getElementById("stat-clusters");

const btnProcess = document.getElementById("btn-process");
const btnExport = document.getElementById("btn-export");
const loadingOverlay = document.getElementById("loading");
const toast = document.getElementById("toast");
const toastMsg = document.getElementById("toast-message");

// 초기화 실행
window.addEventListener("DOMContentLoaded", () => {
    initThreeJS();
    setupEventListeners();
    fetchData(); // 첫 로드 시 실행
});

// 1. Three.js 씬 초기화
function initThreeJS() {
    // --- [왼쪽 뷰포트: Raw Point Cloud] ---
    const containerRaw = document.getElementById("raw-viewport");
    sceneRaw = new THREE.Scene();
    sceneRaw.background = new THREE.Color(0x05070c);

    cameraRaw = new THREE.PerspectiveCamera(60, containerRaw.clientWidth / containerRaw.clientHeight, 0.1, 1000);
    // 카메라 초기 위치 설정 (앞쪽 위에서 내려다보도록)
    cameraRaw.position.set(0, 20, -35); 

    rendererRaw = new THREE.WebGLRenderer({ antialias: true });
    rendererRaw.setSize(containerRaw.clientWidth, containerRaw.clientHeight);
    rendererRaw.setPixelRatio(window.devicePixelRatio);
    containerRaw.appendChild(rendererRaw.domElement);

    controlsRaw = new THREE.OrbitControls(cameraRaw, rendererRaw.domElement);
    controlsRaw.enableDamping = true;
    controlsRaw.dampingFactor = 0.05;
    controlsRaw.target.set(0, -1.73, 20); // 센서 앞쪽 20m 지점을 바라봄

    // --- [오른쪽 뷰포트: Segmented & Clustered] ---
    const containerSeg = document.getElementById("segmented-viewport");
    sceneSeg = new THREE.Scene();
    sceneSeg.background = new THREE.Color(0x05070c);

    cameraSeg = new THREE.PerspectiveCamera(60, containerSeg.clientWidth / containerSeg.clientHeight, 0.1, 1000);
    cameraSeg.position.copy(cameraRaw.position);

    rendererSeg = new THREE.WebGLRenderer({ antialias: true });
    rendererSeg.setSize(containerSeg.clientWidth, containerSeg.clientHeight);
    rendererSeg.setPixelRatio(window.devicePixelRatio);
    containerSeg.appendChild(rendererSeg.domElement);

    controlsSeg = new THREE.OrbitControls(cameraSeg, rendererSeg.domElement);
    controlsSeg.enableDamping = true;
    controlsSeg.dampingFactor = 0.05;
    controlsSeg.target.copy(controlsRaw.target);

    // --- [공통: 그리드 및 좌표축 추가] ---
    // 센서 원점을 기준으로 XZ 평면(라이다 기준 XY평면) 그리드 생성
    const gridHelperRaw = new THREE.GridHelper(80, 80, 0x1e293b, 0x0f172a);
    gridHelperRaw.position.y = -1.73; // 지면에 매칭
    sceneRaw.add(gridHelperRaw);

    const gridHelperSeg = gridHelperRaw.clone();
    sceneSeg.add(gridHelperSeg);

    // 원점 좌표축 (RGB: Red=X_lateral, Green=Y_vertical, Blue=Z_longitudinal)
    // Three.js 좌표계 변환을 반영하여 생성
    const axesRaw = new THREE.AxesHelper(3);
    axesRaw.position.set(0, -1.73, 0);
    sceneRaw.add(axesRaw);

    const axesSeg = axesRaw.clone();
    sceneSeg.add(axesSeg);

    // --- [카메라 동기화 연동] ---
    controlsRaw.addEventListener("change", syncCamerasToSeg);
    controlsSeg.addEventListener("change", syncCamerasToRaw);

    // 애니메이션 루프 가동
    animate();
}

// 2. 카메라 동기화 핸들러
function syncCamerasToSeg() {
    if (isSyncing) return;
    isSyncing = true;
    cameraSeg.position.copy(cameraRaw.position);
    cameraSeg.quaternion.copy(cameraRaw.quaternion);
    controlsSeg.target.copy(controlsRaw.target);
    controlsSeg.update();
    isSyncing = false;
}

function syncCamerasToRaw() {
    if (isSyncing) return;
    isSyncing = true;
    cameraRaw.position.copy(cameraSeg.position);
    cameraRaw.quaternion.copy(cameraSeg.quaternion);
    controlsRaw.target.copy(controlsSeg.target);
    controlsRaw.update();
    isSyncing = false;
}

// 3. 루프 렌더링
function animate() {
    requestAnimationFrame(animate);
    
    controlsRaw.update();
    controlsSeg.update();
    
    rendererRaw.render(sceneRaw, cameraRaw);
    rendererSeg.render(sceneSeg, cameraSeg);
}

// 윈도우 크기 조절 대응
window.addEventListener("resize", () => {
    const containerRaw = document.getElementById("raw-viewport");
    cameraRaw.aspect = containerRaw.clientWidth / containerRaw.clientHeight;
    cameraRaw.updateProjectionMatrix();
    rendererRaw.setSize(containerRaw.clientWidth, containerRaw.clientHeight);

    const containerSeg = document.getElementById("segmented-viewport");
    cameraSeg.aspect = containerSeg.clientWidth / containerSeg.clientHeight;
    cameraSeg.updateProjectionMatrix();
    rendererSeg.setSize(containerSeg.clientWidth, containerSeg.clientHeight);
});

// 4. 슬라이더 이벤트 및 UI 갱신 설정
function setupEventListeners() {
    // 실시간 값 표시 업데이트
    paramVoxel.addEventListener("input", (e) => {
        valVoxel.innerText = parseFloat(e.target.value).toFixed(2);
    });
    paramThreshold.addEventListener("input", (e) => {
        valThreshold.innerText = parseFloat(e.target.value).toFixed(2);
    });
    paramEps.addEventListener("input", (e) => {
        valEps.innerText = parseFloat(e.target.value).toFixed(2);
    });
    paramMinpts.addEventListener("input", (e) => {
        valMinpts.innerText = e.target.value;
    });

    // 업데이트 버튼 클릭
    btnProcess.addEventListener("click", fetchData);

    // JSON 내보내기 버튼 클릭
    btnExport.addEventListener("click", exportObstacleData);
}

// 5. 라이다 데이터 좌표계 변환 헬퍼 함수
// LiDAR 좌표계 (Z-up, X-forward, Y-left/right)
// ➡️ Three.js 좌표계 (Y-up, Z-forward, X-left/right)
// 공식: Three.js X = LiDAR Y (좌우)
//       Three.js Y = LiDAR Z (높이)
//       Three.js Z = LiDAR X (전방)
function convertLidarToThree(point) {
    return new THREE.Vector3(point[1], point[2], point[0]);
}

// 6. API로부터 데이터 받아와서 시각화 요소 배치
function fetchData() {
    loadingOverlay.classList.add("active");

    const voxel = paramVoxel.value;
    const threshold = paramThreshold.value;
    const eps = paramEps.value;
    const minpts = paramMinpts.value;

    const url = `/api/process?distance_threshold=${threshold}&eps=${eps}&min_points=${minpts}&voxel_size=${voxel}`;

    fetch(url)
        .then(res => {
            if (!res.ok) throw new Error("API Process Failed");
            return res.json();
        })
        .then(data => {
            // 통계 수치 갱신
            statTotal.innerText = data.stats.total_points.toLocaleString();
            statGround.innerText = data.stats.ground_points_count.toLocaleString();
            statObstacles.innerText = data.stats.obstacle_points_count.toLocaleString();
            statClusters.innerText = data.stats.num_clusters;
            
            exportData = data.export_data; // 글로벌 데이터에 보관

            // 기존 포인트클라우드 오브젝트 삭제
            clearSceneObjects(sceneRaw);
            clearSceneObjects(sceneSeg);

            // --- [왼쪽 뷰포트 그리기: Raw Point Cloud] ---
            renderRawPointCloud(data.raw_points);

            // --- [오른쪽 뷰포트 그리기: Ground + Clustered Obstacles] ---
            renderSegmentedScene(data);

            loadingOverlay.classList.remove("active");
        })
        .catch(err => {
            console.error(err);
            alert("백엔드 서버로부터 데이터를 가져오는 데 실패했습니다.");
            loadingOverlay.classList.remove("active");
        });
}

// 기존 씬 안의 포인트/박스 메쉬들을 일괄 정리
function clearSceneObjects(scene) {
    const toRemove = [];
    scene.traverse(child => {
        if (child instanceof THREE.Points || child instanceof THREE.LineSegments || child instanceof THREE.Mesh) {
            // 그리드나 축 헬퍼는 제거 대상에서 제외
            if (child.type !== "GridHelper" && child.type !== "AxesHelper") {
                toRemove.push(child);
            }
        }
    });
    toRemove.forEach(obj => {
        scene.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            if (Array.isArray(obj.material)) {
                obj.material.forEach(m => m.dispose());
            } else {
                obj.material.dispose();
            }
        }
    });
}

// 왼쪽 뷰포트에 Raw 포인트 그리기
function renderRawPointCloud(pointsList) {
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    // 거리(깊이)에 따라 멋지게 그라데이션 컬러 매핑
    for (let i = 0; i < pointsList.length; i++) {
        const pt = convertLidarToThree(pointsList[i]);
        positions.push(pt.x, pt.y, pt.z);

        // 라이다 X(전방) 깊이에 따라 색상 변화 (가시성 향상)
        const dist = pointsList[i][0]; // 원래 LiDAR X값
        const factor = Math.min(dist / 40.0, 1.0); // 40m 기준 스케일링
        
        // Neon Blue에서 Cyan으로 이어지는 그라데이션
        colors.push(0.0, 0.5 + (factor * 0.5), 1.0 - (factor * 0.3));
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.12,
        vertexColors: true,
        transparent: true,
        opacity: 0.85
    });

    const pointCloud = new THREE.Points(geometry, material);
    sceneRaw.add(pointCloud);
}

// 오른쪽 뷰포트에 지면 + 군집화된 장애물 + 바운딩 박스 그리기
function renderSegmentedScene(data) {
    // 1. 지면(Road) 포인트 시각화
    if (data.ground_points.length > 0) {
        const groundGeo = new THREE.BufferGeometry();
        const groundPos = [];
        const groundColors = [];

        for (let i = 0; i < data.ground_points.length; i++) {
            const pt = convertLidarToThree(data.ground_points[i]);
            groundPos.push(pt.x, pt.y, pt.z);
            // 차분한 어두운 회색 매핑
            groundColors.push(0.2, 0.22, 0.25);
        }

        groundGeo.setAttribute('position', new THREE.Float32BufferAttribute(groundPos, 3));
        groundGeo.setAttribute('color', new THREE.Float32BufferAttribute(groundColors, 3));

        const groundMat = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            transparent: true,
            opacity: 0.6
        });
        const groundPoints = new THREE.Points(groundGeo, groundMat);
        sceneSeg.add(groundPoints);
    }

    // 2. 장애물 클러스터 시각화
    data.obstacles.forEach(cluster => {
        if (cluster.points.length === 0) return;

        const clusterGeo = new THREE.BufferGeometry();
        const clusterPos = [];
        const clusterColors = [];
        const col = cluster.color; // [r, g, b]

        for (let i = 0; i < cluster.points.length; i++) {
            const pt = convertLidarToThree(cluster.points[i]);
            clusterPos.push(pt.x, pt.y, pt.z);
            clusterColors.push(col[0], col[1], col[2]);
        }

        clusterGeo.setAttribute('position', new THREE.Float32BufferAttribute(clusterPos, 3));
        clusterGeo.setAttribute('color', new THREE.Float32BufferAttribute(clusterColors, 3));

        const clusterMat = new THREE.PointsMaterial({
            size: 0.16, // 장애물을 조금 더 강조해서 크게 그림
            vertexColors: true,
            transparent: false
        });
        const clusterPoints = new THREE.Points(clusterGeo, clusterMat);
        sceneSeg.add(clusterPoints);
    });

    // 3. 노이즈 포인트 시각화
    if (data.noise_points.length > 0) {
        const noiseGeo = new THREE.BufferGeometry();
        const noisePos = [];
        const noiseColors = [];

        for (let i = 0; i < data.noise_points.length; i++) {
            const pt = convertLidarToThree(data.noise_points[i]);
            noisePos.push(pt.x, pt.y, pt.z);
            // 어둡고 탁한 붉은색
            noiseColors.push(0.3, 0.08, 0.08);
        }

        noiseGeo.setAttribute('position', new THREE.Float32BufferAttribute(noisePos, 3));
        noiseGeo.setAttribute('color', new THREE.Float32BufferAttribute(noiseColors, 3));

        const noiseMat = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            transparent: true,
            opacity: 0.5
        });
        const noisePoints = new THREE.Points(noiseGeo, noiseMat);
        sceneSeg.add(noisePoints);
    }

    // 4. Oriented Bounding Box (OBB) 그리기
    data.bboxes.forEach(box => {
        // LiDAR 좌표계 파라미터 로드
        const cx = box.center[0];
        const cy = box.center[1];
        const cz = box.center[2];
        const dx = box.size[0]; // 길이 (Length)
        const dy = box.size[1]; // 너비 (Width)
        const dz = box.size[2]; // 높이 (Height)
        const R = box.R;        // 3x3 회전 행렬

        // 결정론적인 고정 인덱스 순서로 로컬 8개 꼭짓점 생성
        const localCorners = [
            [-dx/2, -dy/2, -dz/2], // 0: 좌-후-하
            [ dx/2, -dy/2, -dz/2], // 1: 우-후-하
            [ dx/2,  dy/2, -dz/2], // 2: 우-전-하
            [-dx/2,  dy/2, -dz/2], // 3: 좌-전-하
            [-dx/2, -dy/2,  dz/2], // 4: 좌-후-상
            [ dx/2, -dy/2,  dz/2], // 5: 우-후-상
            [ dx/2,  dy/2,  dz/2], // 6: 우-전-상
            [-dx/2,  dy/2,  dz/2]  // 7: 좌-전-상
        ];

        // 각 로컬 꼭짓점에 회전행렬 R ➡️ 이동 ➡️ Three.js 좌표계 변환 수행
        const pts = localCorners.map(c => {
            // 1. 3x3 회전 행렬 R 적용 (LiDAR 좌표계 기준)
            const rx = R[0][0]*c[0] + R[0][1]*c[1] + R[0][2]*c[2];
            const ry = R[1][0]*c[0] + R[1][1]*c[1] + R[1][2]*c[2];
            const rz = R[2][0]*c[0] + R[2][1]*c[1] + R[2][2]*c[2];

            // 2. Center 값만큼 평행이동
            const gx = rx + cx;
            const gy = ry + cy;
            const gz = rz + cz;

            // 3. Three.js 좌표계 매핑 (X=LiDAR Y, Y=LiDAR Z, Z=LiDAR X)
            return new THREE.Vector3(gy, gz, gx);
        });

        const col = box.color;
        const colorObj = new THREE.Color(col[0], col[1], col[2]);
        
        // --- 12개 테두리선(LineSegments) 그리기 ---
        const lineIndices = [
            0, 1,  1, 2,  2, 3,  3, 0, // 바닥면 루프
            4, 5,  5, 6,  6, 7,  7, 4, // 윗면 루프
            0, 4,  1, 5,  2, 6,  3, 7  // 수직 기둥들
        ];
        
        const linePositions = [];
        lineIndices.forEach(idx => {
            linePositions.push(pts[idx].x, pts[idx].y, pts[idx].z);
        });
        
        const lineGeo = new THREE.BufferGeometry();
        lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        
        const lineMat = new THREE.LineBasicMaterial({
            color: colorObj,
            linewidth: 2
        });
        
        const boxLines = new THREE.LineSegments(lineGeo, lineMat);
        sceneSeg.add(boxLines);

        // --- 반투명한 3D 면(Mesh) 그리기 ---
        const meshIndices = [
            0, 3, 2,  0, 2, 1, // 바닥면
            4, 5, 6,  4, 6, 7, // 윗면
            0, 1, 5,  0, 5, 4, // 앞면 (0-1-5-4)
            2, 3, 7,  2, 7, 6, // 뒷면 (2-3-7-6)
            3, 0, 4,  3, 4, 7, // 좌측면 (3-0-4-7)
            1, 2, 6,  1, 6, 5  // 우측면 (1-2-6-5)
        ];
        
        const vertices = [];
        pts.forEach(p => {
            vertices.push(p.x, p.y, p.z);
        });
        
        const meshGeo = new THREE.BufferGeometry();
        meshGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        meshGeo.setIndex(meshIndices);
        meshGeo.computeVertexNormals();
        
        const meshMat = new THREE.MeshBasicMaterial({
            color: colorObj,
            transparent: true,
            opacity: 0.08,
            side: THREE.DoubleSide
        });
        
        const boxMesh = new THREE.Mesh(meshGeo, meshMat);
        sceneSeg.add(boxMesh);
    });
}

// 7. 장애물 JSON 데이터 로컬 폴더로 출력 요청
function exportObstacleData() {
    if (!exportData || exportData.length === 0) {
        alert("내보낼 장애물 데이터가 없습니다. 먼저 파라미터를 업데이트하세요.");
        return;
    }

    fetch("/api/export", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(exportData)
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert("에러 발생: " + data.error);
        } else {
            showToast(data.message);
        }
    })
    .catch(err => {
        console.error(err);
        alert("백엔드로 데이터 내보내기 요청 중 오류가 발생했습니다.");
    });
}

// 알림 토스트 출력
function showToast(message) {
    toastMsg.innerText = "Export Successful! saved to stage_2";
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}
