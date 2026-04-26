"""
config.py

项目全局配置文件
通过环境变量控制不同运行模式
"""

import os


class Config:
    """全局配置"""

    # =========================================================================
    # TEE 环境配置
    # =========================================================================
    # 可选值: docker（开发模拟） / tdx（Intel TDX） / sev（AMD SEV）
    TEE_MODE: str = os.environ.get("TEE_MODE", "docker")

    # 是否验证远程证明报告（生产环境应开启）
    VERIFY_ATTESTATION: bool = os.environ.get("VERIFY_ATTESTATION", "false").lower() == "true"

    # =========================================================================
    # gRPC 服务配置
    # =========================================================================
    # 服务监听端口
    GRPC_PORT: int = int(os.environ.get("GRPC_PORT", "50051"))

    # 最大工作线程数
    MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "10"))

    # =========================================================================
    # 证书路径配置
    # =========================================================================
    # 证书目录
    CERTS_DIR: str = os.environ.get("CERTS_DIR", "certs")

    # CA 证书
    CA_CERT: str = os.path.join(CERTS_DIR, "ca.crt")

    # 服务端证书和私钥
    SERVER_CERT: str = os.path.join(CERTS_DIR, "server.crt")
    SERVER_KEY: str = os.path.join(CERTS_DIR, "server.key")

    # 客户端证书和私钥
    CLIENT_CERT: str = os.path.join(CERTS_DIR, "client.crt")
    CLIENT_KEY: str = os.path.join(CERTS_DIR, "client.key")

    # =========================================================================
    # 日志配置
    # =========================================================================
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    # =========================================================================
    # 便利方法
    # =========================================================================
    @classmethod
    def is_development(cls) -> bool:
        return cls.TEE_MODE == "docker"

    @classmethod
    def is_production(cls) -> bool:
        return cls.TEE_MODE in ("tdx", "sev")


# 全局配置单例
config = Config()