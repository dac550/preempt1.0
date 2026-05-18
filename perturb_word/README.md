## 依赖模型下载

首次使用前需下载以下模型（只需执行一次）：

1. Sentence Transformers 模型：
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"

2. FastText 中文模型：
python -c "import fasttext; fasttext.util.download_model('zh', if_exists='ignore')"