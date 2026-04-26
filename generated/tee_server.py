"""
tee_server.py

云端 TEE 服务端，运行在 Gramine/Occlum 保护的 enclave 内。

功能：
    1. 接收本地传来的 FPE 粗脱敏文本和 t2 实体
    2. 执行实体关系识别（预留接口）
    3. 执行语义分析（预留接口）
    4. 返回 edges 和 semantic_info

启动方式：
    python tee_server.py --port 50051
"""

# ===================================用于导入attestation===============================
import sys
import os
sys.path.append(os.path.dirname(__file__))
from tee import create_attestation_provider
# ============================================================================
import argparse
import logging
from concurrent import futures

# ======================================用于导入TLS认证================================
import grpc
from grpc import ssl_server_credentials, ssl_server_certificate_configuration
#============================================================================
# 导入生成的 protobuf 代码
import secure_nlp_pb2 as pb2
import secure_nlp_pb2_grpc as pb2_grpc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# gRPC 服务实现
# ============================================================================

class SecureNLPServicer(pb2_grpc.SecureNLPServiceServicer):
    """
    SecureNLP gRPC 服务实现

    运行在 TEE enclave 内，处理本地传来的请求
    """

    def __init__(self):
        """
        初始化并日志TEE环境（用于 attestation_report输出）
        """
        self.attestation_provider = create_attestation_provider()
        logger.info(f"TEE 方案: {self.attestation_provider.get_name()}")

    def HealthCheck(self, request, context):
        """
        健康检查接口
        """
        logger.info("HealthCheck 被调用")
        return pb2.HealthCheckResponse(
            healthy=True,
            message="TEE SecureNLP Service is running"
        )

    def ProcessInTEE(self, request, context):
        """
        核心处理接口

        接收 FPE 脱敏文本和 t2 实体，返回关系边和语义信息

        TODO: 成员 B 会在这里填充实际的关系识别和语义分析逻辑
        """
        logger.info("=" * 60)
        logger.info(f"ProcessInTEE 被调用")
        logger.info(f"  - 文本长度: {len(request.fpe_sanitized_text)}")
        logger.info(f"  - t2 实体数: {len(request.t2_entities)}")
        logger.info(f"  - epsilon: {request.epsilon}")

        # ================================================================
        # TODO: XJH 在这里填充实际逻辑
        # ================================================================
        #
        # 1. 关系识别（需要填充）
        #    edges = relation_extractor.extract(request.fpe_sanitized_text, request.t2_entities)
        # 2. # 从 TEE 硬件获取真实的证明报告
        # attestation_report = get_tee_attestation()  # 返回真实的加密二进制数据

        # 3. 语义分析（需要填充）
        #    semantic = semantic_analyzer.analyze(request.fpe_sanitized_text)
        #
        # ================================================================
        #
        # 占位返回：空边列表（调试用）
        edges = []
        #=================================================================
        # 占位返回：远程证明报告（）（调试用）
        attestation_report = self.attestation_provider.get_attestation_report()
        #=================================================================
        # 构建响应
        response = pb2.TEEProcessResponse(
            edges=edges,
            #语义部分
            attestation_report=attestation_report
        )
        #=================================================================
        logger.info(f"处理完成: 返回 {len(edges)} 条边")
        logger.info("=" * 60)

        return response


# ============================================================================
# 服务启动
# ============================================================================

def serve(port: int = 50051, max_workers: int = 10):
    """
    启动 gRPC 服务

    Args:
        port: 监听端口
        max_workers: 线程池大小
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    pb2_grpc.add_SecureNLPServiceServicer_to_server(
        SecureNLPServicer(), server
    )

    # 注意：生产环境需要配置 TLS
    # ========================================================================
    # 选择一：开发模式（不使用 TLS，方便本地调试）
    # ========================================================================
    # server.add_insecure_port(f'[::]:{port}')
    # logger.warning("使用 insecure 模式（仅开发环境）")

    # ========================================================================
    # 选择二：生产模式（开启 TLS 1.3 双向认证）
    # ========================================================================
    with open('../certs/server.key', 'rb') as f:
        server_key = f.read()
    with open('../certs/server.crt', 'rb') as f:
        server_cert = f.read()
    with open('../certs/ca.crt', 'rb') as f:
        ca_cert = f.read()

    server_credentials = grpc.ssl_server_credentials(
        ((server_key, server_cert),),
        root_certificates=ca_cert,
        require_client_auth=True
    )

    server.add_secure_port(f'[::]:{port}', server_credentials)
    logger.info("启用 TLS 1.3 双向认证")
    # ========================================================================

    server.start()
    logger.info("服务已就绪，等待请求...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("收到终止信号，正在关闭服务...")
        server.stop(grace=5)
        logger.info("服务已关闭")


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEE SecureNLP 服务")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=50051,
        help="监听端口（默认 50051）"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="工作线程数（默认 10）"
    )

    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)