

# TEE 安全方案设计文档

## 1. 概述

本文档描述了 SecureNLP 系统的可信执行环境（TEE）安全方案。系统采用**环境抽象层**设计，支持从开发阶段的 Docker 模拟环境平滑迁移至生产阶段的硬件 TEE 环境，无需修改业务代码。

### 1.1 设计原则

- **环境无关**：业务代码通过抽象接口与 TEE 环境交互
- **自动适配**：系统自动检测运行环境并选择合适的 TEE 方案
- **零代码迁移**：切换环境仅需修改环境变量，无需改动业务逻辑

### 1.2 架构说明

系统采用**三层架构**实现 TEE 环境抽象：

1. **业务层**：`tee_server.py` 中的业务逻辑，通过抽象接口调用 TEE 功能
2. **抽象层**：`AttestationProvider` 抽象类，定义统一的 TEE 接口
3. **实现层**：
   - `DockerSimulatorProvider`：Docker 模拟环境（当前开发）
   - `IntelTDXProvider`：Intel TDX 硬件 TEE（后期迁移）
   - `AMDSEVProvider`：AMD SEV-SNP 硬件 TEE（后期迁移）

**依赖关系**：业务层 → 抽象层 ← 实现层（工厂模式自动选择）

## 2. 当前方案：Docker 模拟 TEE

### 2.1 方案说明

开发阶段使用 Docker 容器模拟 TEE 环境，提供与真实 TEE 一致的接口，但**不提供硬件级安全保证**。

### 2.2 实现方式

| 组件 | 实现 | 说明 |
|:---|:---|:---|
| 远程证明 | `DockerSimulatorProvider` | 返回模拟的证明报告 |
| 内存安全 | 目前暂不实现（环境依赖） |
| 环境检测 | `is_available()` 返回 `True` | 作为默认兜底方案 |

### 2.3 核心代码

```python
# tee/attestation.py

class DockerSimulatorProvider(AttestationProvider):
    def get_attestation_report(self, user_data: Optional[bytes] = None) -> bytes:
        if user_data:
            return b"DOCKER_SIMULATED_ATTESTATION:" + user_data
        return b"DOCKER_SIMULATED_ATTESTATION"
    
    def is_available(self) -> bool:
        return True
```

### 2.4 使用方式

```bash
# 默认 Docker 模式
python tee_server.py

# 输出日志
[INFO] TEE 方案: DockerSimulatorProvider
```

### 2.5 优缺点

| 优点 | 缺点 |
|:---|:---|
| 零成本，无需特殊硬件 | 无真实安全保证 |
| 开发调试方便 | 远程证明报告为模拟值 |
| 与生产环境接口一致 | 无法演示硬件级安全 |


## 3. 后期方案：硬件 TEE 迁移

### 3.1 方案对比

| 方案 | 类型 | 推荐场景 | 云服务支持 |
|:---|:---|:---|:---|
| **Intel TDX** | 虚拟机级 | 通用计算、AI 推理 | 阿里云 g7t 系列、Azure DCsv3 |
| **AMD SEV-SNP** | 虚拟机级 | 高性能计算 | Azure DCasv5、AWS EC2 |

### 3.2 推荐方案：Intel TDX

#### 3.2.1 选型理由

1. **零代码修改**：现有 Docker 镜像可直接部署
2. **性能损失小**：大型负载下仅 9%-15% 开销
3. **Python 生态友好**：无系统调用限制
4. **云服务可用**：阿里云、Azure 已提供 TDX 实例

#### 3.2.2 部署步骤

```bash
# 1. 创建 TDX 机密计算实例（以阿里云为例）
#    选择 g7t 系列，启用"机密计算"选项

# 2. 安装 Docker
sudo apt update && sudo apt install docker.io

# 3. 设置环境变量
export TEE_MODE=tdx
export VERIFY_ATTESTATION=true

# 4. 拉取并运行镜像
docker run -d -p 50051:50051 \
    -e TEE_MODE=tdx \
    -e VERIFY_ATTESTATION=true \
    secure-nlp-tee:latest
```

#### 3.2.3 待集成内容

| 模块 | 当前状态 | 需要完成的工作 |
|:---|:---|:---|
| `IntelTDXProvider.get_attestation_report()` | 占位 | 集成 Intel TDX SDK，调用 Quote 生成 API |
| `IntelTDXProvider.is_available()` | 已实现 | 检查 `/dev/tdx_guest` 设备 |
| 客户端证明验证 | 未实现 | 集成 TDX 证明验证库 |

### 3.3 备选方案：AMD SEV-SNP

#### 3.3.1 适用场景

- 云服务商主要提供 AMD 实例
- 需要更高扩展性（线程创建效率更高）

#### 3.3.2 切换方式

```bash
export TEE_MODE=sev
python tee_server.py
```

### 3.4 环境切换总结

| 环境 | TEE_MODE | Provider | 远程证明 |
|:---|:---|:---|:---|
| Docker 开发 | `docker` 或不设置 | `DockerSimulatorProvider` | 模拟报告 |
| Intel TDX 生产 | `tdx` | `IntelTDXProvider` | 真实 Quote |
| AMD SEV 生产 | `sev` | `AMDSEVProvider` | 真实 Report |


## 4. 开发注意事项

### 4.1 代码层面

| 注意事项 | 说明 | 当前状态 |
|:---|:---|:---|
| **依赖抽象接口** | 业务代码只调用 `AttestationProvider` 的方法，不直接依赖具体实现 | ✅ 已实现 |
| **环境自动检测** | 使用 `create_attestation_provider()` 工厂方法，不硬编码 Provider | ✅ 已实现 |
| **敏感数据处理** | 在真实 TEE 中应调用 `SecureMemory.wipe()` 擦除敏感数据 | ⏳ 预留接口，待实现 |
| **日志脱敏** | 生产环境避免在日志中输出敏感数据（如原始文本、实体值） | ⚠️ 需注意 |
| **错误处理** | 真实 TEE 环境下 SDK 调用可能失败，需妥善处理异常 | ⚠️ 需完善 |

### 4.2 部署层面

| 注意事项 | 说明 |
|:---|:---|
| **TLS 配置** | 生产环境必须替换 `insecure_channel` 为 `secure_channel` |
| **端口暴露** | TDX 实例需配置安全组，只开放必要端口 |
| **镜像大小** | TEE 实例通常有内存限制，镜像不宜过大 |
| **依赖兼容** | 确认所有 Python 依赖在 TDX 环境中可用 |

### 4.3 测试建议

| 阶段 | 测试内容 | 环境 |
|:---|:---|:---|
| 单元测试 | Provider 接口功能 | Docker |
| 集成测试 | gRPC 通信全链路 | Docker |
| TEE 功能测试 | 远程证明报告获取 | 云 TEE 实例 |
| 性能测试 | 吞吐量、延迟 | 云 TEE 实例 |

### 4.4 迁移检查清单

- [ ] `IntelTDXProvider.get_attestation_report()` 实现真实 Quote 获取
- [ ] 客户端添加证明报告验证逻辑
- [ ] 配置 TLS 双向认证
- [ ] 更新 Dockerfile 添加 TDX SDK 依赖
- [ ] 测试环境变量 `TEE_MODE=tdx` 切换
- [ ] 压测验证性能指标


## 5. 接口说明

### 5.1 AttestationProvider 抽象类

```python
class AttestationProvider(ABC):
    @abstractmethod
    def get_attestation_report(self, user_data: Optional[bytes] = None) -> bytes:
        """获取 TEE 远程证明报告"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查当前环境是否支持该 TEE 方案"""
        pass
```

### 5.2 工厂方法

```python
def create_attestation_provider() -> AttestationProvider:
    """
    自动检测环境，返回合适的 Provider
    
    检测顺序：
    1. 环境变量 TEE_MODE
    2. 检测 /dev/tdx_guest（Intel TDX）
    3. 检测 /dev/sev（AMD SEV）
    4. 默认 DockerSimulatorProvider
    """
```


## 6. 文件结构

```
generated/
├── tee/
│   ├── __init__.py              # 导出公共接口
│   ├── attestation.py           # 远程证明抽象层
│   ├── memory.py                # 安全内存管理（预留）
│   └── providers/
│       ├── __init__.py
│       ├── docker.py            # Docker 模拟（占位）
│       ├── tdx.py               # Intel TDX（占位）
│       └── sev.py               # AMD SEV（占位）
├── tee_server.py                # TEE 服务端
└── local_client.py              # 本地客户端
```


## 7. 参考资料

| 资源 | 链接 |
|:---|:---|
| Intel TDX 官方文档 | https://www.intel.com/tdx |
| AMD SEV-SNP 官方文档 | https://www.amd.com/sev |
| 阿里云机密计算 | https://www.aliyun.com/product/ecs/confidential-computing |
| Azure 机密计算 | https://azure.microsoft.com/confidential-computing |


## 8. 版本记录

| 版本 | 日期 | 作者 | 说明 |
|:---|:---|:---|:---|
| 1.0 | 2026-04-22 | mxy | 初始版本，Docker 模拟 + 后期迁移方案 |

