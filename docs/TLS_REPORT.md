
# TLS 1.3 双向认证 (mTLS) 配置报告

## 1. 概述

SecureNLP 系统本地客户端与云端 TEE 服务之间的通信采用 **TLS 1.3 双向认证（mTLS）** 方案，确保传输层的数据机密性、完整性和身份双向验证。

## 2. 证书体系

本系统采用自建 CA 签发自签名证书的证书体系，结构如下：

| 文件 | 用途 | 持有方 |
|:---|:---|:---|
| `ca.key` | CA 私钥，用于签发服务端和客户端证书 | 安全管理，不传输 |
| `ca.crt` | CA 根证书，用于验证证书签名 | 服务端 + 客户端 |
| `server.key` | 服务端私钥，用于 TLS 握手时证明身份 | 服务端 |
| `server.crt` | 服务端证书，由 CA 签发 | 服务端 |
| `client.key` | 客户端私钥，用于 TLS 握手时证明身份 | 客户端 |
| `client.crt` | 客户端证书，由 CA 签发 | 客户端 |


## 3. 证书参数与生成指令

### 3.1 CA 根证书

**生成指令**：
```powershell
openssl genrsa -out certs/ca.key 2048
openssl req -new -x509 -key certs/ca.key -out certs/ca.crt -days 1825 -subj "/CN=MyProject-CA"
```

**参数说明**：

| 参数 | 值 | 说明 |
|:---|:---|:---|
| 密钥算法 | RSA | 非对称加密算法 |
| 密钥长度 | 2048 位 | 安全性与性能的平衡选择 |
| 证书类型 | 自签名 X.509 | 根证书自签名，无需上级 CA |
| 主题 CN | MyProject-CA | CA 标识名称 |
| 有效期 | 1825 天（5 年） | 长期有效，减少频繁更新 |

**生成文件**：

| 文件 | 说明 |
|:---|:---|
| `certs/ca.key` | CA 私钥，需安全保管 |
| `certs/ca.crt` | CA 根证书，分发至服务端和客户端 |


### 3.2 服务端证书

**生成指令**：
```powershell
openssl genrsa -out certs/server.key 2048
openssl req -new -key certs/server.key -out certs/server.csr -subj "/CN=localhost"
openssl x509 -req -in certs/server.csr -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial -out certs/server.crt -days 365
```

**参数说明**：

| 参数 | 值 | 说明 |
|:---|:---|:---|
| 密钥算法 | RSA | 非对称加密算法 |
| 密钥长度 | 2048 位 | 安全性与性能的平衡选择 |
| 证书类型 | CA 签名 X.509 | 由本系统 CA 签发 |
| 主题 CN | localhost | 必须与客户端连接地址一致 |
| 有效期 | 365 天（1 年） | 定期更新增强安全性 |

**生成文件**：

| 文件 | 说明 |
|:---|:---|
| `certs/server.key` | 服务端私钥 |
| `certs/server.csr` | 证书签名请求（签发后可删除） |
| `certs/server.crt` | 服务端证书 |


### 3.3 客户端证书

**生成指令**：
```powershell
openssl genrsa -out certs/client.key 2048
openssl req -new -key certs/client.key -out certs/client.csr -subj "/CN=MyClient"
openssl x509 -req -in certs/client.csr -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial -out certs/client.crt -days 365
```

**参数说明**：

| 参数 | 值 | 说明 |
|:---|:---|:---|
| 密钥算法 | RSA | 非对称加密算法 |
| 密钥长度 | 2048 位 | 安全性与性能的平衡选择 |
| 证书类型 | CA 签名 X.509 | 由本系统 CA 签发 |
| 主题 CN | MyClient | 客户端标识名称，可自定义 |
| 有效期 | 365 天（1 年） | 定期更新增强安全性 |

**生成文件**：

| 文件 | 说明 |
|:---|:---|
| `certs/client.key` | 客户端私钥 |
| `certs/client.csr` | 证书签名请求（签发后可删除） |
| `certs/client.crt` | 客户端证书 |


## 4. CN 名称与环境对应关系

服务端证书的 CN 必须与客户端连接的地址一致：

| 部署环境 | 客户端连接地址 | 服务端证书 CN | 是否需要重新生成 |
|:---|:---|:---|:---|
| 本机测试 | `localhost:50051` | `localhost` | 否（当前证书适用） |
| Docker 环境 | `tee-server:50051` | `tee-server` | 是 |
| 生产环境 | `tee.example.com:50051` | `tee.example.com` | 是 |

> Docker 环境重新生成服务端证书的指令：
> ```powershell
> openssl req -new -key certs/server.key -out certs/server.csr -subj "/CN=tee-server"
> openssl x509 -req -in certs/server.csr -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial -out certs/server.crt -days 365
> ```


## 5. gRPC TLS 配置

### 5.1 服务端配置

服务端（`tee_server.py`）通过 `grpc.ssl_server_credentials` 启用 TLS，关键配置：

| 配置项 | 值 |
|:---|:---|
| 服务端私钥 | 从 `certs/server.key` 读取 |
| 服务端证书链 | 从 `certs/server.crt` 读取 |
| 客户端验证 CA | 从 `certs/ca.crt` 读取 |
| 是否要求客户端证书 | 是（`require_client_auth=True`） |
| 监听地址 | `[::]:50051`（安全端口） |

### 5.2 客户端配置

客户端（`local_client.py`）通过 `grpc.ssl_channel_credentials` 启用 TLS，关键配置：

| 配置项 | 值 |
|:---|:---|
| 客户端私钥 | 从 `certs/client.key` 读取 |
| 客户端证书链 | 从 `certs/client.crt` 读取 |
| 服务端验证 CA | 从 `certs/ca.crt` 读取 |
| 连接地址 | `localhost:50051`（安全连接） |


## 6. TLS 协议协商

gRPC 基于 HTTP/2 协议，使用 Python `grpc` 库时，TLS 版本由底层 OpenSSL 自动协商。本系统运行环境支持 TLS 1.3，通信时将优先使用 TLS 1.3 协议。

TLS 1.3 相比 TLS 1.2 的优势：

| 特性 | TLS 1.3 |
|:---|:---|
| 握手延迟 | 1-RTT（首次连接），0-RTT（恢复连接） |
| 加密套件 | 仅支持前向安全的 AEAD 加密 |
| 密钥交换 | ECDHE 为主，支持 DHE |


## 7. 安全注意事项

| 注意事项 | 说明 |
|:---|:---|
| CA 私钥保护 | `ca.key` 绝对不可泄露，建议离线保存或使用 HSM |
| 私钥权限 | 生产环境中私钥文件应设置最小权限（如 Linux 下 600） |
| 证书轮换 | 客户端/服务端证书到期前及时更新 |
| Docker 部署 | 证书文件通过 volume 挂载注入，不要硬编码在镜像中 |
| TEE 迁移 | 私钥应存放在 TEE 内部或安全密钥管理系统（如 HSM）中 |


## 8. 证书目录结构

```
项目根目录/
├── certs/
│   ├── ca.key            # CA 私钥（需安全保管）
│   ├── ca.crt            # CA 根证书（分发至服务端和客户端）
│   ├── ca.srl            # 序列号文件（自动生成，勿删除）
│   ├── server.key        # 服务端私钥
│   ├── server.csr        # 服务端签名请求（签发后可删除）
│   ├── server.crt        # 服务端证书
│   ├── client.key        # 客户端私钥
│   ├── client.csr        # 客户端签名请求（签发后可删除）
│   └── client.crt        # 客户端证书
```


