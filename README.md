# preempt1.0
一个隐私信息脱敏系统

## 功能特性

### 双模式脱敏
- **t1 实体（姓名/手机/组织等）**：使用随机姓名替换方案
  - 从预定义的中英文姓名表中随机选择假名进行替换
  - 会话内保持一致：同一姓名在同一会话中映射到相同的假名
  - 跨会话隔离：同一姓名在不同会话中映射到不同的假名
  
- **t2 实体（数值类型）**：使用 mLDP 扰动
  - 对数值进行差分隐私扰动
  - DAG 保证关联数值的数学关系（如单价×数量=总价）

### 姓名表
系统包含以下姓名表：
- `first_names_English.txt`：英文名字表（2000+）
- `last_names_English.txt`：英文姓氏表（4000+）
- `first_names_Chinese.txt`：中文名字表（2000+）
- `last_names_Chinese.txt`：中文姓氏表（400+）

## 使用方法

### 基本使用
```bash
python main.py --input data.txt --epsilon 1.0 --output ./output
```

参数说明：
- `--input`：输入文件路径（默认：data.txt）
- `--epsilon`：隐私预算，越小保护越强（默认：1.0）
- `--output`：输出目录（默认：./output）

### 输出文件
- `sanitized_output.txt`：脱敏后的文本
- `comparison.txt`：原始文本与脱敏文本对照
- `session_infos.json`：会话信息（用于反脱敏）

## 注意事项
每次上传记得关闭SSL验证
