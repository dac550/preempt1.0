"""
name_replacer_module.py — 姓名随机替换模块

设计原则
--------
- 从预定义的中英文姓名表中随机选择姓名进行替换
- 支持中文和英文姓名的识别和替换
- 维护会话内的映射关系，确保同一姓名在同一会话中替换为相同的假名
- 跨会话时，同一姓名会被替换为不同的假名（通过刷新会话实现）
"""

import random
import re
from typing import Dict, Tuple, Optional


class NameReplacer:
    """
    姓名随机替换器
    
    功能：
    - 识别中英文姓名
    - 从姓名表中随机选择替换
    - 维护会话内的映射关系
    """
    
    def __init__(self):
        # 加载姓名表
        self.first_names_english = self._load_name_list("first_names_English.txt")
        self.last_names_english = self._load_name_list("last_names_English.txt")
        self.first_names_chinese = self._load_name_list("first_names_Chinese.txt")
        self.last_names_chinese = self._load_name_list("last_names_Chinese.txt")
        
        # 会话映射表：original_name -> fake_name
        self.session_mapping: Dict[str, str] = {}
        
        # 已使用的假名集合（避免重复）
        self.used_fake_names = set()
        
        # 随机种子（用于会话刷新）
        self.session_seed = random.randint(0, 1000000)
    
    def _load_name_list(self, filename: str) -> list:
        """从文件加载姓名列表"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # 提取列表内容
                if '[' in content and ']' in content:
                    # 使用 eval 解析（简单方式，生产环境建议用 json）
                    import ast
                    # 提取列表部分
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    list_str = content[start:end]
                    return ast.literal_eval(list_str)
                return []
        except Exception as e:
            print(f"加载姓名表 {filename} 失败: {e}")
            return []
    
    def _is_chinese_name(self, name: str) -> bool:
        """判断是否为中文姓名"""
        # 简单判断：包含中文字符
        return bool(re.search(r'[\u4e00-\u9fff]', name))
    
    def _generate_fake_name(self, original_name: str) -> str:
        """生成假名"""
        is_chinese = self._is_chinese_name(original_name)
        
        # 设置随机种子（基于会话种子和原始姓名）
        seed = hash(original_name) + self.session_seed
        rng = random.Random(seed)
        
        max_attempts = 100
        for _ in range(max_attempts):
            if is_chinese:
                # 中文姓名：姓 + 名
                last_name = rng.choice(self.last_names_chinese)
                first_name = rng.choice(self.first_names_chinese)
                fake_name = last_name + first_name
            else:
                # 英文姓名：First Last
                first_name = rng.choice(self.first_names_english)
                last_name = rng.choice(self.last_names_english)
                fake_name = f"{first_name} {last_name}"
            
            # 检查是否已被使用
            if fake_name not in self.used_fake_names:
                self.used_fake_names.add(fake_name)
                return fake_name
        
        # 如果尝试多次仍然重复，直接返回（概率极低）
        return fake_name
    
    def replace_name(self, original_name: str) -> str:
        """
        替换姓名
        
        参数：
            original_name: 原始姓名
        
        返回：
            假名
        """
        # 检查是否已有映射
        if original_name in self.session_mapping:
            return self.session_mapping[original_name]
        
        # 生成新的假名
        fake_name = self._generate_fake_name(original_name)
        
        # 保存映射
        self.session_mapping[original_name] = fake_name
        
        return fake_name
    
    def restore_name(self, fake_name: str) -> Optional[str]:
        """
        还原姓名（反向查找）
        
        参数：
            fake_name: 假名
        
        返回：
            原始姓名，如果找不到则返回 None
        """
        # 反向查找
        for original, fake in self.session_mapping.items():
            if fake == fake_name:
                return original
        return None
    
    def refresh_session(self) -> None:
        """
        刷新会话
        
        清空映射表和已使用假名集合，生成新的会话种子
        确保跨会话时同一姓名会被替换为不同的假名
        """
        self.session_mapping.clear()
        self.used_fake_names.clear()
        self.session_seed = random.randint(0, 1000000)
    
    def get_session_info(self) -> Dict:
        """
        获取会话信息（用于反脱敏）
        
        返回：
            包含映射表的字典
        """
        return {
            "session_seed": self.session_seed,
            "name_mapping": dict(self.session_mapping)
        }


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    replacer = NameReplacer()
    
    # 测试用例
    test_names = [
        "张伟",
        "李明",
        "王芳",
        "Zhang Wei",
        "John Smith",
        "Mary Johnson"
    ]
    
    print("=" * 60)
    print("姓名替换测试")
    print("=" * 60)
    
    # 第一轮会话
    print("\n第一轮会话:")
    for name in test_names:
        fake = replacer.replace_name(name)
        print(f"{name:20} → {fake}")
    
    # 验证同一会话内相同姓名映射一致
    print("\n验证会话内一致性:")
    for name in test_names[:3]:
        fake = replacer.replace_name(name)
        print(f"{name:20} → {fake}")
    
    # 刷新会话
    print("\n" + "=" * 60)
    print("刷新会话")
    print("=" * 60)
    replacer.refresh_session()
    
    # 第二轮会话
    print("\n第二轮会话（同一姓名应映射到不同假名）:")
    for name in test_names:
        fake = replacer.replace_name(name)
        print(f"{name:20} → {fake}")
