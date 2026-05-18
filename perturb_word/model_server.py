"""ST+FT MLDP 扰动服务"""
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface_local'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import socket
import sys
import time
import numpy as np

print("加载ST...")
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("加载FT...")
import fasttext
ft = fasttext.load_model('cc.zh.300.bin')

print("加载词典向量...")
DICT_DIR = 'dict'
dicts = {}
for f in sorted(os.listdir(DICT_DIR)):
    if f.endswith('_words.npy'):
        domain = f.replace('_words.npy', '')
        words = np.load(f'{DICT_DIR}/{f}', allow_pickle=True)
        vecs = np.load(f'{DICT_DIR}/{domain}_vecs.npy')
        dicts[domain] = (words, vecs)
        print(f"  {domain}: {len(words)}词")

print("READY")


def recv_all(conn):
    data = b''
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data.decode('utf-8')


def search_dict_by_vec(vec, domain='medical', top_k=50):
    if domain not in dicts:
        return []
    words, vecs = dicts[domain]
    sims = np.dot(vecs, vec) / (
        np.linalg.norm(vecs, axis=1) * np.linalg.norm(vec) + 1e-8
    )
    idx = np.argsort(-sims)[:top_k]
    return [(words[i], float(sims[i])) for i in idx]


def smart_neighbors(word, epsilon=5.0, k=100, threshold_lower=0.3,
                    threshold_upper=0.95, domain='medical'):
    t_total = time.time()

    # 原词向量
    st_orig = st.encode(word)

    # 加噪声
    noise_scale = 1.0 / epsilon
    noise = np.random.normal(0, noise_scale, st_orig.shape)
    perturbed = st_orig + noise
    print(f"  [噪声] ε={epsilon}, σ={noise_scale:.4f}", file=sys.stderr, flush=True)

    candidates = []

    # 词典搜索（扰动向量）
    dict_cands = search_dict_by_vec(perturbed, domain, top_k=50)
    for cand, sim in dict_cands:
        candidates.append((sim, cand))
    print(f"  [词典] {len(dict_cands)}候选", file=sys.stderr, flush=True)

    # FT搜索（仅短词）
    if len(word) <= 4:
        ft_raw = ft.get_nearest_neighbors(word, k=k)
        for score, near_word in ft_raw:
            if near_word != word:
                candidates.append((score, near_word))
        print(f"  [FT] {len(ft_raw)}候选", file=sys.stderr, flush=True)

    # 去重
    seen = set()
    unique = []
    for score, cand in candidates:
        if cand != word and cand not in seen:
            seen.add(cand)
            unique.append((score, cand))

    if not unique:
        return []

    # 批量编码候选词（只调一次ST）
    cand_words = [c for _, c in unique]
    cand_vecs = st.encode(cand_words, batch_size=32, show_progress_bar=False)

    # 用原词向量过滤（保证语义质量）
    filtered = []
    for i, (score, cand) in enumerate(unique):
        sim = float(np.dot(st_orig, cand_vecs[i]) /
                    (np.linalg.norm(st_orig) * np.linalg.norm(cand_vecs[i]) + 1e-8))
        if threshold_lower <= sim <= threshold_upper:
            filtered.append({'new_word': cand, 'st_sim': sim})

    filtered.sort(key=lambda x: -x['st_sim'])
    print(f"  [完成] {len(filtered)}结果, 耗时{time.time()-t_total:.2f}s",
          file=sys.stderr, flush=True)
    return filtered


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 9999))
server.listen(5)
print("监听9999...", flush=True)

while True:
    conn, addr = server.accept()
    try:
        data = recv_all(conn)
        req = json.loads(data)
        word = req['word']
        epsilon = req.get('epsilon', 5.0)
        domain = req.get('domain', 'medical')
        threshold_lower = req.get('threshold_lower', 0.3)
        threshold_upper = req.get('threshold_upper', 0.95)

        print(f"\n查询: {word} (ε={epsilon})", file=sys.stderr, flush=True)
        result = {
            'word': word,
            'epsilon': epsilon,
            'candidates': smart_neighbors(
                word,
                epsilon=epsilon,
                threshold_lower=threshold_lower,
                threshold_upper=threshold_upper,
                domain=domain
            )
        }
        conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    except Exception as e:
        conn.sendall(json.dumps({'error': str(e)}).encode('utf-8'))
    finally:
        conn.close()
