"""

本地客户端，封装与云端 TEE 服务的 gRPC 通信。

使用方式：
    from generated.local_client import TEEClient

    client = TEEClient("localhost:50051")

    # 健康检查
    if client.health_check():
        print("TEE 服务可用")

    # 发送处理请求
    edges, semantic = client.process_in_tee(
        fpe_sanitized_text="...",
        t2_entities=[...],
        epsilon=1.0
    )
"""

import logging
from typing import List, Tuple, Dict
import grpc



# 导入生成的 protobuf 代码
import secure_nlp_pb2 as pb2
import secure_nlp_pb2_grpc as pb2_grpc


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TEEClient:
    """
    与 TEE 服务的 gRPC 通信逻辑
    """

    def __init__(self, server_address: str = "localhost:50051"):
        """
        初始化客户端

        Args:
            server_address: TEE 服务地址，格式 "host:port"
        """
        self.server_address = server_address  #TEE服务的网络地址
        self.channel = None    #gRPC通信通道（TCP连接，管理连接持，处理网络超时，压缩/解压缩数据）
        self.stub = None   #提供调用远程服务的方法
        self._connect()    #私有方法，用于建立连接

    def _connect(self) -> None:
        """
        建立与 TEE 服务的连接
        """
        try:
            # ============================================================
            # 开发模式（不使用 TLS，方便本地调试）
            # ============================================================
            # self.channel = grpc.insecure_channel(self.server_address)
            # logger.warning("使用 insecure 模式连接 TEE 服务（仅开发环境）")

            # ============================================================
            # 生产模式（TLS 1.3 双向认证）
            # ============================================================
            with open('../certs/client.key', 'rb') as f:
                client_key = f.read()
            with open('../certs/client.crt', 'rb') as f:
                client_cert = f.read()
            with open('../certs/ca.crt', 'rb') as f:
                ca_cert = f.read()

            client_credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert
            )

            self.channel = grpc.secure_channel(self.server_address, client_credentials)
            self.stub = pb2_grpc.SecureNLPServiceStub(self.channel)
            logger.info(f"已建立 TLS 双向安全连接到 TEE 服务: {self.server_address}")
        except Exception as e:
            logger.error(f"连接 TEE 服务失败: {e}")
            raise

    def health_check(self) -> bool:
        """
        检查 TEE 服务是否可用

        Returns:
            True 表示服务正常，False 表示服务异常
        """
        try:
            request = pb2.HealthCheckRequest()
            response = self.stub.HealthCheck(request, timeout=5)
            logger.info(f"TEE 服务状态: healthy={response.healthy}, message={response.message}")
            return response.healthy
        except grpc.RpcError as e:
            logger.error(f"健康检查失败: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            logger.error(f"健康检查异常: {e}")
            return False

    def process_in_tee(
            self,
            fpe_sanitized_text: str,
            t2_entities: List[Tuple[str, str, float, int, int]],
            epsilon: float = 1.0,
            timeout: int = 30
    ) -> Tuple[List[Dict], Dict]:
        """
        将 FPE 粗脱敏数据发送到 TEE 处理

        Args:
            fpe_sanitized_text: FPE 加密后的文本（t1 已脱敏，t2 保持原样）
            t2_entities: t2 实体列表，每个元素为 (token, label, value, start_pos, end_pos)
            epsilon: 隐私预算
            timeout: 超时时间（秒）

        Returns:
            (edges, semantic_info) 元组
            - edges: 关系边列表，每个边为 dict 格式
            - semantic_info: 语义分析结果 dict

        Raises:
            grpc.RpcError: gRPC 调用失败
        """
        # 1. 构建 protobuf 请求
        proto_entities = []
        for token, label, value, start_pos, end_pos in t2_entities:
            proto_entities.append(
                pb2.T2Entity(
                    token=token,
                    label=label,
                    value=value,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
            )

        request = pb2.TEEProcessRequest(
            fpe_sanitized_text=fpe_sanitized_text,
            t2_entities=proto_entities,
            epsilon=epsilon
        )

        logger.info(f"发送 TEE 处理请求: text_len={len(fpe_sanitized_text)}, "
                    f"t2_entities={len(t2_entities)}, epsilon={epsilon}")

        # 2. 调用远程服务
        try:
            response = self.stub.ProcessInTEE(request, timeout=timeout)
        except grpc.RpcError as e:
            logger.error(f"TEE 处理失败: {e.code()} - {e.details()}")
            raise

        # 3. 解析响应
        edges = []
        for edge in response.edges:
            edges.append({
                "from_entity_index": edge.from_entity_index,   #父节点
                "to_entity_index": edge.to_entity_index,  #子节点
                "relation_type": edge.relation_type,    #边关系类型
                "param": edge.param,   #参数
                "has_temp_node": edge.has_temp_node,   #是否有临时节点
                "temp_node_name": edge.temp_node_name,   #临时节点名字
            })

        # ！！！！！！！！注意：当前 proto 中 SemanticAnalysis 被注释了！！！！！！！！！！！
        # 等语义模块实现后，这里需要更新
        semantic_info = {}
        # semantic_info = {
        #     "topic": response.semantic.topic,
        #     "intent": response.semantic.intent,
        #     "keywords": list(response.semantic.keywords),
        #     "embedding": list(response.semantic.embedding),
        #     "context_summary": response.semantic.context_summary,
        # }

        # 远程证明报告（可选，用于验证 TEE 完整性和安全性）
        attestation = response.attestation_report
        if attestation:
            logger.info(f"收到远程证明报告: {len(attestation)} bytes")

        logger.info(f"TEE 处理完成: 收到 {len(edges)} 条边")

        return edges, semantic_info

    def close(self) -> None:
        """关闭连接"""
        if self.channel:
            self.channel.close()
            logger.info("TEE 客户端连接已关闭")

    def __enter__(self):
        """支持 with 语句，自动关闭连接，避免忘记调用close()"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句时自动关闭连接"""
        self.close()


# ---------------------------------------------------------------------------
# 测试代码
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TEE 客户端测试")
    print("=" * 60)

    # 创建客户端
    client = TEEClient("localhost:50051")

    try:
        # 1. 健康检查
        print("\n[1] 健康检查...")
        if not client.health_check():
            print("❌ TEE 服务不可用，请先启动 tee_server.py")
            exit(1)
        print("✅ TEE 服务正常")

        # 2. 测试处理请求
        print("\n[2] 发送处理请求...")

        # 模拟数据
        fpe_text = "In 2024, customer Regan Anthony purchased 3 items, each priced at 150 yuan, total 450 yuan."

        t2_entities = [
            ("2024", "YEAR", 2024.0, 3, 7),
            ("3", "COUNT", 3.0, 42, 43),
            ("150", "AMOUNT", 150.0, 62, 65),
            ("450", "AMOUNT", 450.0, 72, 75),
        ]

        edges, semantic = client.process_in_tee(fpe_text, t2_entities, epsilon=1.0)

        print(f"✅ 收到 {len(edges)} 条边")
        for i, edge in enumerate(edges):
            print(f"   边 {i + 1}: {edge}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    finally:
        client.close()
        print("\n测试完成")