D:\Desktop\mytask\
│
├── local_service/                        # 本地服务（主入口）
│   ├── main.py                           # 主入口：编排全流程
│   ├── local_sanitizer.py                # 阶段1（NER+t1脱敏）+ 阶段3（DAG+mLDP）
│   ├── api.py                            # NER 识别（调用云端大模型 API）
│   ├── dag_module.py                     # DAG 构建 + mLDP 扰动
│   ├── mLDP_module.py                    # 指数机制差分隐私扰动
│   ├── ff3_module.py                     # FPE 格式保持加密
│   ├── name_replacer_module.py           # 姓名随机替换
│   ├── entity_types.py                   # t1/t2 实体类型判断
│   ├── tee_contract.py                   # 通信数据结构定义（Payload 类）
│   ├── data.txt                          # 测试输入文本
│   ├── local_output.json                 # 本地预处理输出（离线模式用）
│   ├── final_output.json                 # 最终脱敏结果输出
│   ├── first_names_Chinese.txt           # 中文名字库
│   ├── first_names_English.txt           # 英文名字库
│   ├── last_names_Chinese.txt            # 中文姓氏库
│   ├── last_names_English.txt            # 英文姓氏库
│   └── README.md                         # 本地服务说明
│
├── generated/                            # gRPC 通信基础设施
│   ├── __init__.py                       # 导出 TEEClient + proto 类
│   ├── secure_nlp.proto                  # Protobuf 接口定义（源文件）
│   ├── secure_nlp_pb2.py                 # 消息类（自动生成）
│   ├── secure_nlp_pb2_grpc.py            # 服务类（自动生成）
│   ├── local_client.py                   # gRPC 客户端封装
│   ├── tee_server.py                     # gRPC 服务端（TEE 侧）
│   └── tee/                              # TEE 抽象层
│       ├── __init__.py                   # TEE 抽象层入口
│       ├── attestation.py                # 远程证明（Docker + TDX/SEV 预留）
│       ├── memory.py                     # 安全内存管理（预留）
│       └── providers/                    # 各 TEE 方案实现（预留）
│           ├── __init__.py
│           ├── docker.py
│           ├── tdx.py
│           └── sev.py
│
├── certs/                                # TLS 证书
│   ├── ca.key                            # CA 私钥（安全保管）
│   ├── ca.crt                            # CA 根证书
│   ├── server.key                        # 服务端私钥
│   ├── server.crt                        # 服务端证书
│   ├── client.key                        # 客户端私钥
│   └── client.crt                        # 客户端证书
│
├── config.py                             # 配置文件（待完善）
├── requirements.txt                      # Python 依赖清单
├── Dockerfile                            # Docker 镜像构建文件（待完善）
├── docker-compose.yml                    # Docker 编排配置（待完善）
│
└── docs/                                 # 文档
    ├── architecture.md                   # 代码架构文档（待创建）
    ├── TEE_Security_Solution.md          # TEE 安全方案文档
    ├── TLS_Configuration_Report.md       # TLS 配置报告
    └── Docker_Configuration_Report.md    # Docker 配置报告