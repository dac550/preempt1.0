"""
tee 抽象层

提供与 TEE 环境交互的统一接口，支持：
- 远程证明（Attestation）
- 安全内存管理（Secure Memory）

使用方式：
    from tee import create_attestation_provider, SecureMemory

    # 自动检测当前环境并返回合适的 Provider
    attestation = create_attestation_provider()
    report = attestation.get_attestation_report()

    # 安全擦除敏感数据
    SecureMemory.wipe(sensitive_data)
"""

from .attestation import (
    AttestationProvider,
    DockerSimulatorProvider,
    IntelTDXProvider,
    AMDSEVProvider,
    create_attestation_provider,
)

#from .memory import (
#   SecureMemory,
#    SecureContext,
#)

__all__ = [
    # 远程证明
    "AttestationProvider",
    "DockerSimulatorProvider",
    "IntelTDXProvider",
    "AMDSEVProvider",
    "create_attestation_provider",
    # 内存安全
    #"SecureMemory",
    #"SecureContext",
]