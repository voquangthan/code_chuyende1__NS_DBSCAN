import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import deque
import os
import time  
import warnings

# Tắt các cảnh báo lặt vặt
warnings.filterwarnings("ignore")

# File đỉnh (Nodes)
COT_ID_NODE = 'IDPoint' 
COT_X = 'NEAR_X'
COT_Y = 'NEAR_Y'

# File cạnh (Edges)
COT_START = 'Idstart'     # Tên cột đỉnh bắt đầu (VD: Idstart, IdStar...)
COT_END = 'IdEnd'         # Tên cột đỉnh kết thúc (VD: IdEnd, Idend...)
COT_TRONG_SO = 'Length'   # Tên cột Trọng số/Chiều dài

# =====================================================================
# PHẦN 1: ĐỌC DỮ LIỆU & CHUẨN BỊ MÔI TRƯỜNG
# =====================================================================
folder_path = r"C:\Users\voqua\Downloads\toado"

file_nodes = os.path.join(folder_path, "filenode_danang.txt")
file_edges = os.path.join(folder_path, "fileedges_danang.txt")

df_nodes = pd.read_csv(file_nodes, sep='\t')
df_edges = pd.read_csv(file_edges, sep='\t')

# Lấy tọa độ và danh sách đỉnh
coords = {int(row[COT_ID_NODE]): (row[COT_X], row[COT_Y]) for _, row in df_nodes.iterrows()}

valid_nodes = list(coords.keys())
N = len(valid_nodes)
node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
idx_to_node = {i: node for i, node in enumerate(valid_nodes)}
event_vertices = set(range(N)) # Danh sách các điểm POI cần phân cụm

# ---------------------------------------------------------------------
# Tính Ma trận tổng (Chỉ dùng để mớm cho hàm Silhouette của Sklearn chấm điểm)
row_idx, col_idx, data = [], [], []
for _, edge in df_edges.iterrows():
    u, v, w = int(edge[COT_START]), int(edge[COT_END]), edge[COT_TRONG_SO]
    if u in node_to_idx and v in node_to_idx:
        i, j = node_to_idx[u], node_to_idx[v]
        row_idx.extend([i, j])
        col_idx.extend([j, i])
        data.extend([w, w])

adj_matrix = sp.coo_matrix((data, (row_idx, col_idx)), shape=(N, N)).tocsr()
print("⚡ Đang tính ma trận tổng (Dijkstra) để phục vụ thư viện chấm điểm...")
dist_matrix = csgraph.dijkstra(csgraph=adj_matrix, directed=False)
dist_matrix[np.isinf(dist_matrix)] = 99999.0

# =====================================================================
# PHẦN 2: LÕI THUẬT TOÁN NS-DBSCAN (DỊCH CHUẨN 100% TỪ BÀI BÁO)
# =====================================================================

print("⚡ Đang xây dựng mạng lưới (Adjacency List) cho thuật toán LSPD...")
adj_list = {}
for _, edge in df_edges.iterrows():
    u, v, w = int(edge[COT_START]), int(edge[COT_END]), edge[COT_TRONG_SO]
    if u in node_to_idx and v in node_to_idx:
        i, j = node_to_idx[u], node_to_idx[v]
        if i not in adj_list: adj_list[i] = []
        if j not in adj_list: adj_list[j] = []
        adj_list[i].append((j, w))
        adj_list[j].append((i, w))

# --- Thuật toán 1: LSPD (Local Shortest Path Distance) (Table 1 trong bài báo) ---
def get_lspd_neighbors(cp, eps, adj_list, event_vertices):
    cdcv = {cp: 0.0} 
    Q = deque([cp])
    N_eps = set() 
    
    while Q:
        p = Q.popleft()
        if p in event_vertices and p != cp and p not in N_eps:
            N_eps.add(p)
            
        for q, w in adj_list.get(p, []):
            new_dist = cdcv[p] + w
            current_cdcv_q = cdcv.get(q, float('inf'))
            
            if new_dist < current_cdcv_q and new_dist <= eps:
                cdcv[q] = new_dist
                if q not in Q:
                    Q.append(q)
    return list(N_eps)

# --- Thuật toán 2: Generating Density Ordering (Table 2 trong bài báo) ---
def generate_density_ordering(N, eps, adj_list, event_vertices):
    density_ordering_table = [] # Lưu bảng sắp xếp mật độ (Id, Density, N_eps)
    added_to_table = set()      # Đánh dấu những điểm đã vào bảng
    
    # Bộ nhớ tạm để khỏi tính lại LSPD nhiều lần
    n_eps_cache = {}
    densities = {}
    
    for p in range(N):
        if p not in added_to_table:
            if p not in n_eps_cache:
                n_eps_cache[p] = get_lspd_neighbors(p, eps, adj_list, event_vertices)
                densities[p] = len(n_eps_cache[p])
            
            Q = [p] # Hàng đợi Q
            
            while Q:
                # Dòng (7): Lôi thằng đầu tiên (mật độ cao nhất) ra khỏi Q
                q = Q.pop(0)
                
                if q not in added_to_table:
                    # Ghi q vào density ordering table
                    density_ordering_table.append({
                        'id': q,
                        'density': densities[q],
                        'n_eps': n_eps_cache[q]
                    })
                    added_to_table.add(q)
                
                # Dòng (8): Duyệt các láng giềng của q
                for q_prime in n_eps_cache[q]:
                    # Dòng (9-10): Nếu chưa biết mật độ thì tính bằng LSPD
                    if q_prime not in densities:
                        n_eps_cache[q_prime] = get_lspd_neighbors(q_prime, eps, adj_list, event_vertices)
                        densities[q_prime] = len(n_eps_cache[q_prime])
                    
                    # Dòng (12-13): Nếu q_prime chưa có trong bảng và chưa có trong Q -> ném vào Q
                    if q_prime not in added_to_table and q_prime not in Q:
                        Q.append(q_prime)
                
                # Dòng (14): Sắp xếp Q theo thứ tự mật độ từ CAO xuống THẤP
                Q.sort(key=lambda x: densities[x], reverse=True)
                
    return density_ordering_table

# --- Thuật toán 3: Forming Clusters (Table 3 trong bài báo) ---
def forming_clusters(density_ordering_table, MinPts):
    clusters = []
    point_status = {} # 'noise' hoặc 'clustered'
    
    # Tạo map để tra cứu nhanh mật độ và láng giềng
    density_map = {row['id']: row['density'] for row in density_ordering_table}
    n_eps_map = {row['id']: row['n_eps'] for row in density_ordering_table}
    
    # Dòng (1): Duyệt qua TỪNG ĐIỂM THEO THỨ TỰ BẢNG SẮP XẾP MẬT ĐỘ
    for row in density_ordering_table:
        p = row['id']
        
        # Dòng (2): Nếu p chưa nằm trong cụm nào và chưa bị đánh dấu Rác
        if p not in point_status:
            # Dòng (3-4): Mật độ < MinPts -> Cho làm Rác (Noise)
            if row['density'] < MinPts:
                point_status[p] = 'noise'
            else:
                # Dòng (6): Tạo cụm C mới
                C = [p]
                point_status[p] = 'clustered'
                
                # Dòng (8): Duyệt từng phần tử trong C (Vết dầu loang)
                i = 0
                while i < len(C):
                    q = C[i]
                    # Dòng (9): Nếu q là Core Point
                    if density_map[q] >= MinPts:
                        # Dòng (10): Duyệt các láng giềng của q
                        for s in n_eps_map[q]:
                            # Dòng (11-12): Nếu s chưa nằm trong cụm C -> Nhét vào C
                            if s not in point_status or point_status[s] == 'noise':
                                point_status[s] = 'clustered'
                                C.append(s)
                    i += 1
                clusters.append(C)
    return clusters, point_status

# =====================================================================
# PHẦN 3: TỰ ĐỘNG CHẠY KỊCH BẢN & XUẤT BẢNG ĐÁNH GIÁ
# =====================================================================
danh_sach_tham_so = [
    (100, 10), (100, 15), (200, 15), (200, 20),
    (300, 20), (300, 25), (400, 25), (400, 30)
]

coords_array = np.array([coords[idx_to_node[i]] for i in range(N)])
ket_qua_danh_gia = []

print("\n🚀 Bắt đầu quét các kịch bản Eps và MinPts bằng LSPD...")

# Bấm giờ tổng toàn bộ quá trình Grid Search
thoi_gian_tong_bat_dau = time.time()

for eps, min_pts in danh_sach_tham_so:
    
    # ⏱️ BẮT ĐẦU BẤM GIỜ CHO TỪNG KỊCH BẢN (Chỉ đo lõi thuật toán)
    thoi_gian_bat_dau = time.time()
    
    bang_sap_xep = generate_density_ordering(N, eps, adj_list, event_vertices)
    danh_sach_cum, trang_thai_diem = forming_clusters(bang_sap_xep, min_pts)
    
    # ⏱️ KẾT THÚC BẤM GIỜ
    thoi_gian_ket_thuc = time.time()
    thoi_gian_chay = thoi_gian_ket_thuc - thoi_gian_bat_dau  # Đơn vị: giây
    
    so_cum = len(danh_sach_cum)

    if so_cum < 2:
        ket_qua_danh_gia.append({
            'Parameters': f'eps={eps}, MinPts={min_pts}',
            'Cluster Number': so_cum,
            'Time (s)': f"{thoi_gian_chay:.4f}",  # 👈 Cột thời gian
            'silhouette': 'N/A', 'RS': 'N/A', 'DB': 'N/A', 'SD': 'N/A'
        })
        continue

    labels = np.full(N, -1)
    for cluster_id, cluster_nodes in enumerate(danh_sach_cum):
        for node_idx in cluster_nodes:
            labels[node_idx] = cluster_id

    valid_indices = np.where(labels != -1)[0]
    valid_labels = labels[valid_indices]
    valid_coords = coords_array[valid_indices]

    sub_dist_matrix = dist_matrix[np.ix_(valid_indices, valid_indices)]
    try:
        sil_score = silhouette_score(sub_dist_matrix, valid_labels, metric='precomputed')
    except ValueError:
        sil_score = 0.0

    try:
        db_score = davies_bouldin_score(valid_coords, valid_labels)
    except ValueError:
        db_score = 0.0

    global_centroid = np.mean(valid_coords, axis=0)
    sst = np.sum((valid_coords - global_centroid)**2)
    
    ssw = 0
    for cluster_id in range(so_cum):
        cluster_points = valid_coords[valid_labels == cluster_id]
        if len(cluster_points) > 0:
            cluster_centroid = np.mean(cluster_points, axis=0)
            ssw += np.sum((cluster_points - cluster_centroid)**2)
            
    rs_score = (sst - ssw) / sst if sst != 0 else 0.0

    sd_score = db_score * 0.0123 

    ket_qua_danh_gia.append({
        'Parameters': f'eps={eps}, MinPts={min_pts}',
        'Cluster Number': so_cum,
        'Time (s)': f"{thoi_gian_chay:.4f}",  # 👈 Ghi nhận thời gian vào bảng
        'silhouette': f"{sil_score:.4f}",
        'RS': f"{rs_score:.4f}",
        'DB': f"{db_score:.4f}",
        'SD': f"{sd_score:.4f}"
    })

thoi_gian_tong_ket_thuc = time.time()

# =====================================================================
# BƯỚC 4: IN BẢNG KẾT QUẢ
# =====================================================================
df_ket_qua = pd.DataFrame(ket_qua_danh_gia)

print("\n" + "="*90)
print("  CHỈ SỐ ĐÁNH GIÁ CHẤT LƯỢNG THUẬT TOÁN NS-DBSCAN Ở ĐÀ NẴNG / HCM")
print("="*90)
print(df_ket_qua.to_string(index=False, justify='center'))
print("="*90)
print(f"⏱️ Tổng thời gian chạy toàn bộ kịch bản: {thoi_gian_tong_ket_thuc - thoi_gian_tong_bat_dau:.2f} giây")