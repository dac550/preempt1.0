"""BioConceptVec 服务 - LLM多候选翻译 + BCV词表验证"""
import os
import socket
import json
import numpy as np
from gensim.models import KeyedVectors
from openai import OpenAI

LLM_API_KEY = "821a645b91c04ba4ab1b2db01b050f32.Zfw4xlprX362QxIJ"
llm = OpenAI(api_key=LLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")

print("加载 BioConceptVec...")
bv = KeyedVectors.load_word2vec_format(
    os.path.expanduser('~/bioconceptvec_word2vec_skipgram.bin'),
    binary=True
)
print("READY")


def map_to_bcv(zh_word):
    """中文→多候选英文翻译→BCV词表验证"""
    resp = llm.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": f"""将中文医学术语"{zh_word}"翻译成英文。
要求：
1. 输出3个候选翻译，从最简到最完整，每行一个，只输出英文
2. 示例：
   尿路感染：
   uti
   urinary infection
   urinary tract infection
3. 术语：{zh_word}"""}],
        temperature=0.1,
        max_tokens=64
    )
    
    candidates = resp.choices[0].message.content.strip().split('\n')
    candidates = [c.strip().lower() for c in candidates if c.strip()]
    
    # 逐个检查
    for c in candidates:
        if c in bv:
            return c
        underscore = c.replace(' ', '_')
        if underscore in bv:
            return underscore
    
    # 模糊搜索
    first_word = candidates[0].split()[0] if candidates else ''
    if first_word:
        matches = [w for w in bv.key_to_index
                   if first_word in w.lower() and len(w) < 30
                   and w.replace('-', '').replace('_', '').isalpha()]
        if matches:
            return min(matches, key=len)
    
    return None


def llm_translate_en_zh(text):
    resp = llm.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": f"将以下医学术语翻译成中文，只输出中文：{text}"}],
        temperature=0.1,
        max_tokens=64
    )
    return resp.choices[0].message.content.strip()


def analogical_perturb(root_orig_zh, root_pert_zh, child_zh, top_k=10):
    root_orig_en = map_to_bcv(root_orig_zh)
    root_pert_en = map_to_bcv(root_pert_zh)
    child_en = map_to_bcv(child_zh)

    failed = []
    if not root_orig_en: failed.append(root_orig_zh)
    if not root_pert_en: failed.append(root_pert_zh)
    if not child_en: failed.append(child_zh)

    if failed:
        return {'error': '映射失败', 'failed': failed}

    v_root_orig = bv[root_orig_en]
    v_root_pert = bv[root_pert_en]
    v_child = bv[child_en]
    delta = v_root_pert - v_root_orig
    v_result = v_child + delta
    neighbors = bv.similar_by_vector(v_result, topn=top_k)

    results = []
    for word, sim in neighbors:
        zh = llm_translate_en_zh(word)
        results.append({'bcv_word': word, 'zh_word': zh, 'similarity': float(sim)})

    return {
        'mapping': {root_orig_zh: root_orig_en, root_pert_zh: root_pert_en, child_zh: child_en},
        'results': results
    }


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 9997))
server.listen(5)
print("监听 9997...", flush=True)


def recv_all(conn):
    data = b''
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data.decode('utf-8')


while True:
    conn, addr = server.accept()
    try:
        data = recv_all(conn)
        req = json.loads(data)
        result = analogical_perturb(
            req['root_orig'], req['root_pert'], req['child_orig'],
            req.get('top_k', 10)
        )
        conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    except Exception as e:
        conn.sendall(json.dumps({'error': str(e)}).encode('utf-8'))
    finally:
        conn.close()
