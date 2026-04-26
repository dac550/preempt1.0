"""
远程证明（Remote Attestation）抽象层

提供统一的接口来获取 TEE 远程证明报告。
支持自动检测当前运行环境，并返回合适的 Provider。

支持的环境：
- Docker 模拟环境（开发测试） *（目前只关注这个）
- Intel TDX（后续）
- AMD SEV-SNP（后续）
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


# ============================================================================
# 抽象基类
# ============================================================================

class AttestationProvider(ABC):
    """远程证明提供者抽象接口"""

    @abstractmethod
    def get_attestation_report(self, user_data: Optional[bytes] = None) -> bytes:
        """
        获取 TEE 远程证明报告

        Args:
            user_data: 可选的自定义数据，会被绑定到报告中（签名）（用于防止重放攻击）

        Returns:
            加密的证明报告（bytes）

        Raises:
            RuntimeError: 当前环境不支持该 TEE 方案或获取报告失败
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        检查当前环境是否支持该 TEE 方案

        Returns:
            True 表示可用，False 表示不可用
        """
        pass

    def get_name(self) -> str:
        """返回 Provider 名称"""
        return self.__class__.__name__


# ============================================================================
# Docker 模拟环境（开发测试用）*
# ============================================================================

class DockerSimulatorProvider(AttestationProvider):
    """
    Docker 模拟环境

    用于开发测试，返回模拟的证明报告。
    不提供任何真实的安全保证。
    """

    def get_attestation_report(self, user_data: Optional[bytes] = None) -> bytes:
        """返回模拟的证明报告"""
        # 如果提供了 user_data，可以将其包含在模拟报告中
        if user_data:
            return b"DOCKER_SIMULATED_ATTESTATION:" + user_data
        return b"DOCKER_SIMULATED_ATTESTATION"

    def is_available(self) -> bool:
        """Docker 模拟环境始终可用"""
        return True
# ============================================================================
#  后续环境都先不用看（目前）
# ============================================================================
# Intel TDX 真实环境（后续）
# ============================================================================

class IntelTDXProvider(AttestationProvider):
    """
    Intel TDX 真实环境

    需要运行在支持 Intel TDX 的云实例上。
    依赖 Intel TDX SDK 或 /dev/tdx_guest 设备。
    """

    def get_attestation_report(self, user_data: Optional[bytes] = None) -> bytes:
        """
        获取 Intel TDX Quote

        TODO: 集成 Intel TDX SDK
        参考文档: https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html
        """
        # 检查是否真的在 TDX 环境中
        if not self.is_available():
            raise RuntimeError("当前环境不支持 Intel TDX")

        # TODO: 实际实现
        # from tdx_attestation import get_quote
        # return get_quote(user_data)

        raise NotImplementedError(
            "Intel TDX SDK 尚未集成。"
            "请参考 Intel TDX 文档集成 TDX Attestation API。"
        )

    def is_available(self) -> bool:
        """检查是否运行在 Intel TDX 环境中"""
        # 检查 TDX Guest 设备是否存在
        return os.path.exists("/dev/tdx_guest")


# ============================================================================
# AMD SEV-SNP 真实环境
# ============================================================================

class AMDSEVProvider(AttestationProvider):
    """
    AMD SEV-SNP 真实环境

    需要运行在支持 AMD SEV-SNP 的云实例上。
    依赖 AMD SEV SDK 或 /dev/sev 设备。
    """

    def get_attestation_report(self, user_data: Optional[bytes] = None) -> bytes:
        """
        获取 AMD SEV-SNP Attestation Report

        TODO: 集成 AMD SEV SDK
        参考文档: https://www.amd.com/en/developer/sev.html
        """
        if not self.is_available():
            raise RuntimeError("当前环境不支持 AMD SEV-SNP")

        # TODO: 实际实现
        # from sev_attestation import get_attestation_report
        # return get_attestation_report(user_data)

        raise NotImplementedError(
            "AMD SEV SDK 尚未集成。"
            "请参考 AMD SEV 文档集成 SEV Attestation API。"
        )

    def is_available(self) -> bool:
        """检查是否运行在 AMD SEV 环境中"""
        return os.path.exists("/dev/sev")


# ============================================================================
# 工厂方法：自动检测环境
# ============================================================================

def create_attestation_provider() -> AttestationProvider:
    """
    自动检测当前环境，返回合适的远程证明提供者

    检测顺序：
    1. 检查环境变量 TEE_MODE 是否强制指定
    2. 检测 Intel TDX
    3. 检测 AMD SEV
    4. 默认使用 Docker 模拟器

    Returns:
        适用于当前环境的 AttestationProvider 实例
    """
    # 1. 检查环境变量覆盖
    env_override = os.environ.get("TEE_MODE", "").lower()

    if env_override == "docker":
        return DockerSimulatorProvider()
    elif env_override == "tdx":
        return IntelTDXProvider()
    elif env_override == "sev":
        return AMDSEVProvider()

    # 2. 自动检测
    providers = [
        IntelTDXProvider(),
        AMDSEVProvider(),
    ]

    for provider in providers:
        if provider.is_available():
            return provider

    # 3. 默认返回 Docker 模拟器
    return DockerSimulatorProvider()