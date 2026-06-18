import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from collections import deque
import heapq
import os
import time
import warnings
import math

warnings.filterwarnings("ignore")

# ==========================================================
# CẤU HÌNH DỮ LIỆU
# ==========================================================
COT_ID_NODE  = 'IDPoint'
COT_X        = 'NEAR_X'
COT_Y        = 'NEAR_Y'
COT_START    = 'Idstart'
COT_END      = 'IdEnd'
COT_TRONG_SO = 'Length'

folder_path = r"C:\Users\voqua\Downloads\dulieu"
file_nodes  = os.path.join(folder_path, "filenode_HCM.txt")
file_edges  = os.path.join(folder_path, "fileedges_HCM.txt")

# ==========================================================
# ĐỌC DỮ LIỆU & XÂY DỰNG ĐỒ THỊ
# ==========================================================
df_nodes = pd.read_csv(file_nodes, sep='\t')
df_edges = pd.read_csv(file_edges, sep='\t')

coords = {int(row[COT_ID_NODE]): (row[COT_X], row[COT_Y]) for _, row in df_nodes.iterrows()}

valid_nodes   = list(coords.keys())
N             = len(valid_nodes)
node_to_idx   = {node: i for i, node in enumerate(valid_nodes)}
idx_to_node   = {i: node for i, node in enumerate(valid_nodes)}
coords_array  = np.array([coords[idx_to_node[i]] for i in range(N)])
event_vertices = set(range(N))

row_idx, col_idx, data = [], [], []
adj_list = {}

for _, edge in df_edges.iterrows():
    u, v, w = int(edge[COT_START]), int(edge[COT_END]), edge[COT_TRONG_SO]
    if u in node_to_idx and v in node_to_idx:
        i, j = node_to_idx[u], node_to_idx[v]
        row_idx.extend([i, j])
        col_idx.extend([j, i])
        data.extend([w, w])
        adj_list.setdefault(i, []).append((j, w))
        adj_list.setdefault(j, []).append((i, w))

adj_matrix = sp.coo_matrix((data, (row_idx, col_idx)), shape=(N, N)).tocsr()
print("⚡ Đang chạy Dijkstra toàn cục (Chỉ dùng để chấm điểm Silhouette, không dùng phân cụm)...")
dist_matrix = csgraph.dijkstra(csgraph=adj_matrix, directed=False)
dist_matrix[np.isinf(dist_matrix)] = 99999.0

# ==========================================================
# CẢI TIẾN 1: LSPD DIJKSTRA (CHỐT KHOẢNG CÁCH CHUẨN)
# ==========================================================
def get_neighbors(cp: int, eps: float):
    cdcv = {cp: 0.0}
    heap = [(0.0, cp)]
    N_eps = set()

    while heap:
        dist_p, p = heapq.heappop(heap)
        if dist_p > cdcv.get(p, 1e18):
            continue

        if p in event_vertices and p != cp:
            N_eps.add(p)

        for q, w in adj_list.get(p, []):
            nd = cdcv[p] + w
            if nd <= eps and nd < cdcv.get(q, 1e18):
                cdcv[q] = nd
                heapq.heappush(heap, (nd, q))

    return list(N_eps), cdcv

# ==========================================================
# CẢI TIẾN 2: GAUSSIAN WEIGHTED DENSITY (MẬT ĐỘ TRỌNG SỐ)
# ==========================================================
def weighted_density(neighbors, cdcv, h):
    rho = 0.0
    for q in neighbors:
        d = cdcv[q]
        rho += math.exp(-d / h)
    return rho

# ==========================================================
# CẢI TIẾN 3: ADAPTIVE EPS (BÁN KÍNH THÍCH NGHI KẸP BIÊN)
# ==========================================================
def adaptive_eps(base_eps: float, rho: float, rho_mean: float, gamma: float = 0.3):
    ratio = rho_mean / (rho + 1e-6)
    eps_v = base_eps * (ratio ** gamma)
    return max(base_eps * 0.6, min(base_eps * 1.4, eps_v))

# ==========================================================
# TẠO CỤM (BẢN CHUẨN: CHỈ DÙNG 1 ĐIỀU KIỆN LÀ MẬT ĐỘ)
# ==========================================================
def cluster_forming(order_ids: list, densities: dict, n_eps_cache: dict, minpts_count: float):
    labels = np.full(N, -1)
    cid    = 0

    for p in order_ids:
        if labels[p] != -1: continue
            
        # Điều kiện điểm lõi: Mật độ Gaussian >= MinPts
        if densities[p] < minpts_count:
            continue

        labels[p] = cid
        Q = deque([p])

        while Q:
            q = Q.popleft()
            for s in n_eps_cache.get(q, []):
                if labels[s] == -1:
                    labels[s] = cid
                    # Nếu láng giềng cũng là lõi thì lan truyền tiếp
                    if densities[s] >= minpts_count:
                        Q.append(s)
        cid += 1
    return labels

# ==========================================================
# HÀM TÍNH TOÁN ĐỘ PHÂN TÁN (SD)
# ==========================================================
def calc_sd(coords, labels):
    cluster_ids = set(labels) - {-1}
    vals = []
    for cid in cluster_ids:
        pts = coords[labels == cid]
        if len(pts) < 2: continue
        center = pts.mean(axis=0)
        vals.append(np.mean(np.linalg.norm(pts - center, axis=1)))
    return np.mean(vals) if vals else 0.0

# ==========================================================
# BƯỚC CUỐI: CHẠY AUTO-TUNING VÀ ĐÁNH GIÁ THỜI GIAN
# ==========================================================
eps_params    = [100, 150, 200, 250, 300, 350, 400]
minpts_params = [10, 15, 20, 25, 30] 
results       = []

print("\n🚀 Chạy AWD-NS-DBSCAN (Bản 3 Cải tiến Cốt lõi)...\n")

thoi_gian_tong_bat_dau = time.time()

for base_eps in eps_params:
    
    # ⏱️ BẤM GIỜ TỪNG KỊCH BẢN EPS (Vì Eps ảnh hưởng thời gian chạy nhiều nhất)
    thoi_gian_bat_dau = time.time()
    
    rho0 = {}
    for p in range(N):
        nb, cdcv = get_neighbors(p, base_eps)
        rho0[p]  = weighted_density(nb, cdcv, base_eps)

    rho_mean = np.mean(list(rho0.values()))
    densities, n_eps_cache = {}, {}

    for p in range(N):
        eps_v          = adaptive_eps(base_eps, rho0[p], rho_mean)
        nb, cdcv       = get_neighbors(p, eps_v)
        n_eps_cache[p] = nb
        densities[p]   = weighted_density(nb, cdcv, eps_v)

    # Đã gỡ bỏ Max-heap, dùng hàm sort cơ bản của Python (nhanh tương đương, không mang tính phô trương)
    order_ids = sorted(range(N), key=lambda x: densities[x], reverse=True)

    for minpts_val in minpts_params:
        
        # Đã gỡ bỏ ràng buộc kép và hút điểm rác (refine_boundary)
        labels = cluster_forming(order_ids, densities, n_eps_cache, minpts_count=minpts_val)

        # ⏱️ CHỐT THỜI GIAN LÕI THUẬT TOÁN
        thoi_gian_ket_thuc = time.time()
        thoi_gian_chay = thoi_gian_ket_thuc - thoi_gian_bat_dau

        n_cluster = len(set(labels) - {-1})
        if n_cluster < 2: continue

        valid               = np.where(labels != -1)[0]
        valid_labels        = labels[valid]
        valid_coords        = coords_array[valid]
        sub_dist            = dist_matrix[np.ix_(valid, valid)]

        # Chuẩn hóa dữ liệu trước khi chấm điểm DB và SD
        scaler              = StandardScaler()
        valid_coords_scaled = scaler.fit_transform(valid_coords)

        try:
            sil = silhouette_score(sub_dist, valid_labels, metric='precomputed')
            db = davies_bouldin_score(valid_coords_scaled, valid_labels)
        except ValueError:
            sil, db = 0.0, 0.0

        cs = (1 / (1 + db)) + sil
        sd = calc_sd(valid_coords_scaled, valid_labels)

        results.append({
            "eps_base":   base_eps,
            "minpts":     minpts_val,
            "clusters":   n_cluster,
            "Time(s)":    round(thoi_gian_chay, 4), # 👈 Cột thời gian chạy
            "silhouette": round(sil, 4),
            "CS":         round(cs,  4),
            "DB":         round(db,  4),
            "SD":         round(sd,  4),
        })

df_valid = pd.DataFrame(results)
thoi_gian_tong_ket_thuc = time.time()

if not df_valid.empty:
    for col in ['silhouette', 'CS', 'DB', 'SD']:
        min_v, max_v = df_valid[col].min(), df_valid[col].max()
        df_valid[f'{col}_n'] = 0.5 if max_v == min_v else (df_valid[col] - min_v) / (max_v - min_v)

    df_valid['J_Score'] = df_valid['silhouette_n'] + df_valid['CS_n'] - df_valid['DB_n'] - df_valid['SD_n']
    best_scenario = df_valid.loc[df_valid['J_Score'].idxmax()]

    print(df_valid[['eps_base', 'minpts', 'clusters', 'Time(s)', 'silhouette', 'CS', 'DB', 'SD']].to_string(index=False))
    print(f"\n🏆 AUTO-TUNING: Kịch bản Đỉnh Nhất là eps_base={best_scenario['eps_base']}, minpts={best_scenario['minpts']}")
    print(f"⏱️ Tổng thời gian chạy toàn bộ kịch bản: {thoi_gian_tong_ket_thuc - thoi_gian_tong_bat_dau:.2f} giây")
else:
    print(" Không tìm thấy cụm hợp lệ!")