# local_service

本地侧服务，只负责：

- 调用 `NERAPI` 识别所有标签。
- 对 t1 标签执行姓名替换或 FPE 加密。
- 保留 t2 原始 token、label、数值和原文位置。
- 按 `secure_nlp.proto` 中的 `TEEProcessRequest` 形态组装请求数据。

本地侧不执行 DAG 构建；DAG/关系边由远程侧返回后，本地侧再调用复制过来的 `dag_module.py` 和 `mLDP_module.py` 完成 t2 扰动加密。

运行示例：

```bash
python main.py --input ../data.txt --epsilon 1.0 --output local_output.json
```

拿到远程侧返回后，在本地完成 t2 扰动：

```bash
python main.py --input local_output.json --remote-response ../remote_service/remote_output.json --final-output final_output.json
```
