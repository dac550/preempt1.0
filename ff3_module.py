"""
ff3_module.py — 自适应多语言格式保持加密

修复说明
--------
Bug1 (Padding截断):
  加密时保存完整 padded 密文到 enc_meta，解密时直接对完整密文解密再截取原长。
  不再在密文侧截断，只在解密结果侧截取。

Bug2 (跨段不对齐):
  段元数据从 (block_start, cat_prefix, seg_len) 改为
  (block_start, cat_prefix, orig_len, enc_full)
  enc_full 是完整 padded 密文，解密时直接使用，无需重建。

设计原则
--------
- 从 token 字符所在 Unicode block 动态构建字母表，语言无关
- 每个「连续同质段」（同 block + 同大类）独立用对应字母表的 FF3 加密
- 标点/空格通过 struct 原样保留，不参与加密
- 字符集严格保持：加密后每个字符仍在原 Unicode block 内
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ff3 import FF3Cipher

_FF3_MIN_LEN = 6
_BLOCK_SIZE  = 256


# ---------------------------------------------------------------------------
# 字母表构建
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _block_alphabet(block_start: int, cat_prefix: str) -> str:
    """
    构建指定 block 内的字母表。
    cat_prefix:
      "LA" → 仅 ASCII 字母（A-Za-z），用于基础拉丁字符
      "L"  → 仅非 ASCII 字母（拉丁补充、西里尔、平假名等）
      "D"  → 不走此函数，由调用方直接返回 _DIGIT_ALPHABET
    """
    if cat_prefix == "LA":
        return ''.join(
            chr(cp)
            for cp in range(block_start, block_start + _BLOCK_SIZE)
            if unicodedata.category(chr(cp)).startswith("L") and chr(cp).isascii()
        )
    # cat_prefix == "L"：非 ASCII 字母
    return ''.join(
        chr(cp)
        for cp in range(block_start, block_start + _BLOCK_SIZE)
        if unicodedata.category(chr(cp)).startswith("L") and not chr(cp).isascii()
    )


def _get_block_start(ch: str) -> int:
    return (ord(ch) // _BLOCK_SIZE) * _BLOCK_SIZE


_DIGIT_SENTINEL = -1   # 数字段的虚拟 block_start，与真实 block 不冲突
_DIGIT_ALPHABET = "0123456789"

def _char_cat_prefix(ch: str) -> Optional[str]:
    """
    字符大类：
      'D'  → ASCII 十进制数字 (0-9)，固定字母表 "0123456789"
      'LA' → ASCII 字母 (A-Za-z)，字母表仅含 A-Za-z，不混入拉丁补充
      'L'  → 非 ASCII Unicode 字母（德语 äöü、西里尔、平假名等），走 block 动态构建
      None → 标点/空格等，不参与加密
    """
    if ch.isdigit() and ch.isascii():
        return "D"
    if unicodedata.category(ch).startswith("L"):
        return "LA" if ch.isascii() else "L"
    return None


# ---------------------------------------------------------------------------
# 段切分
# ---------------------------------------------------------------------------

class Segment:
    __slots__ = ("chars", "block_start", "cat_prefix")

    def __init__(self, chars: str, block_start: int, cat_prefix: str):
        self.chars       = chars
        self.block_start = block_start
        self.cat_prefix  = cat_prefix

    @property
    def alphabet(self) -> str:
        if self.cat_prefix == "D":
            return _DIGIT_ALPHABET
        return _block_alphabet(self.block_start, self.cat_prefix)

    def is_valid(self) -> bool:
        return len(self.alphabet) >= 2


def _split_segments(token: str) -> Tuple[List[str], List[Segment]]:
    struct:   List[str]     = []
    segments: List[Segment] = []

    cur_chars:  List[str]      = []
    cur_block:  Optional[int]  = None
    cur_cat:    Optional[str]  = None

    def flush():
        nonlocal cur_block, cur_cat
        if cur_chars:
            segments.append(Segment(''.join(cur_chars), cur_block, cur_cat))
        cur_chars.clear()
        cur_block = None
        cur_cat   = None

    for ch in token:
        cp = _char_cat_prefix(ch)
        if cp is None:
            flush()
            struct.append(ch)
        else:
            bs = _DIGIT_SENTINEL if cp == "D" else _get_block_start(ch)
            if bs == cur_block and cp == cur_cat:
                cur_chars.append(ch)
            else:
                flush()
                cur_chars.append(ch)
                cur_block = bs
                cur_cat   = cp
            struct.append('O')

    flush()
    return struct, segments


# ---------------------------------------------------------------------------
# FPEManager
# ---------------------------------------------------------------------------

class FPEManager:
    """
    格式保持加密管理器。

    key   : 长期密钥，用户持久保存，跨会话不变
    tweak : 会话密钥，每轮对话随机生成，用完即换

    相同 key + 不同 tweak → 相同明文映射到不同密文
    → 服务商无法跨对话关联同一个实体
    """

    def __init__(self, key: Optional[str] = None, tweak: Optional[str] = None):
        self.key   = key   if key   else os.urandom(16).hex().upper()
        self.tweak = tweak if tweak else os.urandom(7).hex().upper()
        self._cipher_cache: Dict[str, FF3Cipher] = {}

    def refresh_session_tweak(self) -> str:
        """
        生成新的会话 tweak，清空 cipher 缓存。
        每轮对话开始时调用，保证同一实体在不同对话里密文不同。
        返回新 tweak 供调用方记录（用于本次会话的反脱敏）。
        """
        self.tweak = os.urandom(7).hex().upper()
        self._cipher_cache.clear()   # 旧 cipher 全部失效
        return self.tweak

    def _get_cipher(self, alphabet: str) -> FF3Cipher:
        if alphabet not in self._cipher_cache:
            if alphabet == _DIGIT_ALPHABET:
                # 数字专用：用 radix=10，FF3 原生支持
                self._cipher_cache[alphabet] = FF3Cipher(
                    self.key, self.tweak, radix=10)
            else:
                self._cipher_cache[alphabet] = FF3Cipher.withCustomAlphabet(
                    self.key, self.tweak, alphabet)
        return self._cipher_cache[alphabet]

    def _encrypt_seg(self, seg: Segment) -> Tuple[str, str]:
        """
        返回 (enc_display, enc_full)：
          enc_display : 截取原长，用于拼回密文字符串
          enc_full    : 完整 padded 密文，存入 enc_meta 供解密使用
        """
        alphabet = seg.alphabet
        cipher   = self._get_cipher(alphabet)
        text     = seg.chars
        orig_len = len(text)
        pad_len  = max(0, _FF3_MIN_LEN - orig_len)

        padded = text + alphabet[0] * pad_len
        enc_full = cipher.encrypt(padded)          # 完整加密

        return enc_full[:orig_len], enc_full        # display 截取，full 保留

    def _decrypt_seg(self, enc_full: str, orig_len: int,
                     block_start: int, cat_prefix: str) -> str:
        """用完整 padded 密文解密，截取前 orig_len 位。"""
        if cat_prefix == "D":
            alphabet = _DIGIT_ALPHABET
        else:
            alphabet = _block_alphabet(block_start, cat_prefix)
        cipher   = self._get_cipher(alphabet)
        dec_full = cipher.decrypt(enc_full)
        return dec_full[:orig_len]

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def encrypt_master(self, token: str) -> Tuple[str, List]:
        """
        Returns
        -------
        ciphertext : 加密结果字符串（有效字符仍在原 Unicode block）
        enc_meta   : [struct, [(block_start, cat_prefix, orig_len, enc_full), ...]]
        """
        if not token:
            return token, [[], []]

        struct, segments = _split_segments(token)

        display_parts: List[str] = []
        seg_meta: List[Tuple[int, str, int, str]] = []

        for seg in segments:
            if seg.is_valid():
                enc_display, enc_full = self._encrypt_seg(seg)
            else:
                enc_display = seg.chars
                enc_full    = seg.chars
            display_parts.append(enc_display)
            seg_meta.append((seg.block_start, seg.cat_prefix,
                             len(seg.chars), enc_full))

        # 按 struct 填回密文字符
        enc_iter = iter(''.join(display_parts))
        result   = [next(enc_iter) if ch == 'O' else ch for ch in struct]

        return ''.join(result), [struct, seg_meta]

    def decrypt_master(self, ciphertext: str, enc_meta: List) -> str:
        """
        解密。enc_meta 是 encrypt_master 返回的第二个值。
        解密时直接使用 enc_full（无需重建 padded 密文），避免 padding 截断问题。
        """
        if not ciphertext or not enc_meta or not enc_meta[0]:
            return ciphertext

        struct, seg_meta = enc_meta

        dec_parts: List[str] = []
        for block_start, cat_prefix, orig_len, enc_full in seg_meta:
            dec = self._decrypt_seg(enc_full, orig_len, block_start, cat_prefix)
            dec_parts.append(dec)

        # 按 struct 填回解密字符
        dec_iter = iter(''.join(dec_parts))
        result   = [next(dec_iter) if ch == 'O' else ch for ch in struct]

        return ''.join(result)

    def get_key_material(self) -> dict:
        return {"key": self.key, "tweak": self.tweak}


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fpe = FPEManager()

    cases = [
        "Zhang Wei", "A", "AB",
        "Müller", "Gänzlich", "Ön", "über", "straße", "Ü",
        "Jean-François", "María",
        "411-+888a", "13800138000",
        "Иванов",
        "やまだ",
    ]

    print(f"{'原始':<20} {'加密结果':<28} {'解密一致':<8} {'字符集严格保持'}")
    print("-" * 78)
    for token in cases:
        enc, meta = fpe.encrypt_master(token)
        dec       = fpe.decrypt_master(enc, meta)
        ok        = "✓" if dec == token else f"✗ got {dec!r}"

        # 严格验证：cat_prefix 相同（ASCII字母不会变成拉丁补充字母）
        charset_ok = all(
            _char_cat_prefix(o) == _char_cat_prefix(e)
            for o, e in zip(token, enc)
            if _char_cat_prefix(o) is not None
        )
        cs = "✓" if charset_ok else "✗"

        print(f"{token!r:<20} {enc!r:<28} {ok:<8} {cs}")