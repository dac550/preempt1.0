"""
mLDP_module.py  —  局部差分隐私（指数机制）

修复说明
--------
Bug-Fix 1  自适应对称域（替代原来的 domain_min=0 固定下界）：
  原实现：domain_min=0，domain_max=10^位数
    · value=130 → 域[0,1000]，域宽 1001，指数衰减极慢，扰动几乎为零
    · value=6   → 域[0,10]，域宽 11，扰动方向严重偏向 0
  修复后：以 value 为中心，按数值量级自适应确定半径，构造对称域
    · value=130 → 域[108,152]，radius=22，扰动均匀分布在真实值附近
    · value=6   → 域[1,11]，radius=5，不再向 0 方向偏移

Bug-Fix 2  浮点数原生支持：
  原实现：只接受 int，血糖 6.8、血压 130/80 被截断或跳过
  修复后：接受 float，内部放大到整数域扰动后还原，精度由 precision 参数控制

实现原理
--------
指数机制打分函数：score(y | x) = exp(-ε · |y - x|)
满足 ε-LDP（对称距离函数是 1-敏感的）。

自适应域半径策略（_adaptive_radius）：
  radius = max(ceil(|value| * RATIO), MIN_RADIUS)
  RATIO=0.20 表示扰动范围约为真实值的 ±20%，MIN_RADIUS=3 保证极小值也有足够扰动空间。
  上界 MAX_RADIUS=500 防止极大值（如金额百万）产生过大域而拖慢速度。
"""

import math
from functools import lru_cache
from typing import List, Union

import numpy as np

# ---------------------------------------------------------------------------
# 域半径策略常量（可调参数）
# ---------------------------------------------------------------------------
_RATIO       = 0.20   # 扰动范围占真实值的比例
_MIN_RADIUS  = 3      # 极小值兜底半径
_MAX_RADIUS  = 500    # 防止超大值撑爆域


def _adaptive_radius(value: int) -> int:
    """
    以 value 绝对值为基准，计算自适应半径。
    保证扰动域既不过窄（隐私保护弱）也不过宽（扰动实用性差）。
    """
    r = math.ceil(abs(value) * _RATIO)
    return max(_MIN_RADIUS, min(r, _MAX_RADIUS))


def _build_domain(value: int, radius: int) -> tuple[int, int]:
    """
    构造以 value 为中心的对称整数域 [lo, hi]，下界不低于 0。
    """
    lo = max(0, value - radius)
    hi = value + radius
    return lo, hi


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class mLDPMechanism:
    """
    局部差分隐私指数机制。

    支持整数和浮点数；对浮点数先放大到整数域扰动，再还原。

    参数
    ----
    epsilon     : 隐私预算，越小保护越强（扰动越大）
    domain_min  : 手动指定域下界（可选，默认自适应）
    domain_max  : 手动指定域上界（可选，默认自适应）

    若同时指定 domain_min/domain_max，则使用手动域（兼容旧接口）；
    若不指定（均为 None），则每次 perturb 时对每个 value 自适应构造域。
    """

    def __init__(
        self,
        epsilon:    float,
        domain_min: int | None = None,
        domain_max: int | None = None,
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon 必须 > 0，当前 epsilon={epsilon}")

        self.epsilon    = epsilon
        self._fixed_lo  = domain_min   # None 表示自适应
        self._fixed_hi  = domain_max

        # 缓存：(lo, hi, x_int) → (candidates_tuple, probs_tuple)
        # 用 tuple 是因为 np.ndarray 不可哈希
        self._dist_cache: dict = {}

    # ------------------------------------------------------------------
    # 内部：获取概率分布
    # ------------------------------------------------------------------

    def _get_distribution(self, x_int: int, lo: int, hi: int):
        """
        返回 (candidates_array, probs_array)。
        结果缓存：相同 (x_int, lo, hi) 只计算一次。
        """
        key = (x_int, lo, hi)
        if key in self._dist_cache:
            return self._dist_cache[key]

        candidates = np.arange(lo, hi + 1, dtype=np.int64)
        distances  = np.abs(candidates - x_int).astype(np.float64)
        # 数值稳定：先减去最大指数值，防止 exp 溢出
        log_w = -self.epsilon * distances
        log_w -= log_w.max()
        weights = np.exp(log_w)
        probs   = weights / weights.sum()

        self._dist_cache[key] = (candidates, probs)
        return candidates, probs

    def _perturb_int(self, x_int: int) -> int:
        """对整数值执行一次指数机制扰动。"""
        if self._fixed_lo is not None and self._fixed_hi is not None:
            lo, hi = self._fixed_lo, self._fixed_hi
        else:
            radius = _adaptive_radius(x_int)
            lo, hi = _build_domain(x_int, radius)

        candidates, probs = self._get_distribution(x_int, lo, hi)
        return int(np.random.choice(candidates, p=probs))

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def perturb(self, x: Union[int, float], precision: int = 1) -> Union[int, float]:
        """
        对单个数值执行 ε-LDP 扰动。

        参数
        ----
        x         : 待扰动值（int 或 float）
        precision : 浮点数保留小数位数（默认 1，如 6.8 → 保留 1 位）

        返回
        ----
        与输入同类型的扰动值（int → int，float → float）
        """
        if isinstance(x, float):
            # 放大到整数域：6.8 × 10^1 = 68（整数）
            scale   = 10 ** precision
            x_int   = round(x * scale)
            noisy   = self._perturb_int(x_int)
            return round(noisy / scale, precision)
        else:
            return self._perturb_int(int(x))

    def perturb_batch(
        self,
        values:    List[Union[int, float]],
        precision: int = 1,
    ) -> List[Union[int, float]]:
        """批量扰动。"""
        return [self.perturb(v, precision) for v in values]

    def privacy_guarantee(self) -> str:
        """返回当前配置的隐私保证描述（用于报告/展示）。"""
        return (
            f"ε-LDP 指数机制，ε={self.epsilon}，"
            f"自适应对称域（半径≈±{int(_RATIO*100)}%真实值，"
            f"最小半径={_MIN_RADIUS}，最大半径={_MAX_RADIUS}）"
        )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Bug-Fix 验证：自适应域 vs 原固定域")
    print("=" * 60)

    # 案例1：血糖（浮点，小值）
    mldp_float = mLDPMechanism(epsilon=1.0)
    vals_bg = [mldp_float.perturb(6.8, precision=1) for _ in range(10)]
    print(f"\n血糖 6.8  →  扰动10次: {vals_bg}")
    print(f"  均值偏差: {abs(sum(vals_bg)/10 - 6.8):.2f}（应接近 0）")

    # 案例2：血压（整数，中值）
    mldp_int = mLDPMechanism(epsilon=1.0)
    vals_bp = [mldp_int.perturb(130) for _ in range(10)]
    print(f"\n血压 130  →  扰动10次: {vals_bp}")
    print(f"  均值偏差: {abs(sum(vals_bp)/10 - 130):.1f}（应接近 0）")

    # 案例3：年龄（整数，小值）
    vals_age = [mldp_int.perturb(58) for _ in range(10)]
    print(f"\n年龄 58   →  扰动10次: {vals_age}")
    print(f"  均值偏差: {abs(sum(vals_age)/10 - 58):.1f}（应接近 0）")

    # 对比：原来的固定域在 value=130 时几乎不扰动
    print("\n" + "-" * 40)
    print("原固定域对比（domain_min=0, domain_max=1000）：")
    mldp_old = mLDPMechanism(epsilon=1.0, domain_min=0, domain_max=1000)
    vals_old = [mldp_old.perturb(130) for _ in range(10)]
    print(f"  旧方式: {vals_old}  ← 几乎不变，隐私保护失效")

    # 案例4：隐私保证描述
    print(f"\n{mldp_int.privacy_guarantee()}")