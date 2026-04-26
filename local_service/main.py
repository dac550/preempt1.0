import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from generated import TEEClient

import argparse
import json
from pathlib import Path
from local_sanitizer import LocalPreprocessor
from tee_contract import RelationEdgePayload


def main():
    #  =======================================命令行解析=================================
    parser = argparse.ArgumentParser(description="本地预处理服务：NER + t1 加密 + TEE 请求组装")
    parser.add_argument("--input", "-i", default="../data.txt", help="输入文本文件路径")
    parser.add_argument("--epsilon", "-e", type=float, default=1.0, help="隐私预算")
    parser.add_argument("--output", "-o", default="./local_output.json", help="输出请求 JSON 路径")
    parser.add_argument("--remote-response", default=None, help="远程侧返回的 DAG/edges JSON")
    parser.add_argument("--final-output", default="./final_output.json", help="本地 t2 加密后的最终输出路径")
    args = parser.parse_args()
    # =================================================================================
    # ======================== gRPC 调用云端 TEE ========================
    input_path = Path(args.input)
    texts = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    preprocessor = LocalPreprocessor(epsilon=args.epsilon)

    # 创建 gRPC 客户端
    client = TEEClient("localhost:50051")

    # 健康检查
    if not client.health_check():
        print("无法连接到 TEE 服务，请先启动 tee_server.py")
        client.close()
        return

    final_records = []
    try:
        for text in texts:
            # 阶段1：本地预处理
            request, session_info = preprocessor.prepare_remote_request(text)

            # 阶段2：gRPC 调用云端
            t2_entities = [
                (e.token, e.label, e.value, e.start_pos, e.end_pos)
                for e in request.t2_entities
            ]
            edges, _ = client.process_in_tee(
                fpe_sanitized_text=request.fpe_sanitized_text,
                t2_entities=t2_entities,
                epsilon=request.epsilon
            )

            # 阶段3：本地后处理
            relation_edges = [
                RelationEdgePayload(
                    from_entity_index=edge["from_entity_index"],
                    to_entity_index=edge["to_entity_index"],
                    relation_type=edge["relation_type"],
                    param=edge["param"],
                    has_temp_node=edge.get("has_temp_node", False),
                    temp_node_name=edge.get("temp_node_name", ""),
                )
                for edge in edges
            ]
            sanitized_text, dag_info = preprocessor.apply_remote_dag(request, relation_edges)

            final_records.append({
                "original_text": text,
                "sanitized_text": sanitized_text,
                "dag_info": dag_info,
                "local_session": session_info,
            })
            print(f" 处理完成")

        # 保存最终结果
        Path(args.final_output).write_text(
            json.dumps(final_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"全流程完成，最终输出: {args.final_output}")

    finally:
        client.close()


if __name__ == "__main__":
    main()