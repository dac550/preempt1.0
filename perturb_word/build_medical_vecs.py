"""编码医疗词表为 npy，供 model_server.py 加载"""
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface_local'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
from sentence_transformers import SentenceTransformer

print("加载 ST...")
st = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', local_files_only=True)

print("加载医疗词表...")
with open('dict/THUOCL_medical.txt', 'r', encoding='utf-8') as f:
    words = [line.strip().split('\t')[0] for line in f if line.strip()]
words = [w for w in words if len(w) >= 2]
print(f"有效词: {len(words)}")

print("编码...")
vecs = st.encode(words, batch_size=64, show_progress_bar=True)

np.save('dict/medical_words.npy', np.array(words))
np.save('dict/medical_vecs.npy', vecs)
print("✅ 保存完成")
