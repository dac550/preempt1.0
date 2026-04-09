# 姓名加密方案变更说明

## 变更概述
将姓名的 FPE（格式保持加密）方案改为从预定义姓名表中随机选择并替换的方案。

## 主要变更

### 1. 新增模块
- **name_replacer_module.py**：姓名随机替换模块
  - 从中英文姓名表中随机选择假名
  - 维护会话内的映射关系
  - 支持跨会话隔离

### 2. 修改模块
- **sanitizer_module.py**：
  - 移除 FPE 加密相关代码
  - 使用 NameReplacer 替代 FPEManager
  - 更新 session_info 结构（移除 session_tweak，添加 name_session）
  - 简化 get_key_info() 为 get_session_info()

- **main.py**：
  - 移除密钥生成和保存相关代码
  - 简化初始化流程

- **README.md**：
  - 更新功能说明
  - 添加姓名表说明

### 3. 新增测试
- **test_name_replacement.py**：完整的姓名替换功能测试

## 方案对比

### 原方案（FPE 加密）
**优点：**
- 格式保持：加密后字符仍在原 Unicode block
- 可逆性强：通过密钥可以精确还原
- 理论安全性高

**缺点：**
- 需要管理长期密钥（key）和会话密钥（tweak）
- 密文可能不自然（如 "Zhang Wei" → "Xbkqp Wfj"）
- 实现复杂度高

### 新方案（随机替换）
**优点：**
- 替换结果自然：假名都是真实的姓名
- 无需密钥管理：不需要保存长期密钥
- 实现简单：易于理解和维护
- 可读性好：脱敏后的文本更易阅读

**缺点：**
- 需要维护姓名表
- 理论上可能出现假名重复（概率极低）

## 功能保持

以下功能保持不变：
- ✓ 会话内一致性：同一姓名在同一会话中映射到相同假名
- ✓ 跨会话隔离：同一姓名在不同会话中映射到不同假名
- ✓ 反脱敏功能：可以通过 session_info 还原原始姓名
- ✓ t2 实体处理：数值扰动逻辑完全不变
- ✓ DAG 关系保持：关联数值的数学关系保持不变

## 使用示例

### 中文姓名
```
原始: 我叫张伟，今年25岁
脱敏: 我叫却泽恩，今年25.0岁
还原: 我叫张伟，今年25.0岁
```

### 英文姓名
```
原始: My name is John Smith
脱敏: My name is Calvin Mcbride
还原: My name is John Smith
```

### 混合场景
```
原始: Zhang Wei purchased 3 items, each priced at 150 yuan
脱敏: Dennis Mcbride purchased 4.0 items, each priced at 154.0 yuan
还原: Zhang Wei purchased 4.0 items, each priced at 154.0 yuan
```

## 测试验证

运行测试：
```bash
# 测试姓名替换模块
python name_replacer_module.py

# 测试完整脱敏系统
python sanitizer_module.py

# 运行完整测试套件
python test_name_replacement.py

# 测试实际使用场景
python main.py --input data.txt --epsilon 1.0 --output ./output
```

所有测试均已通过 ✓
