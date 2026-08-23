# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.posttraining - 后训练全生命周期 (SFT -> RLHF -> DPO) 对齐工程与能力演进模型

包含：
- `AlignmentPipeline`: 模型在预训练、SFT、RLHF 与 DPO 四大阶段下的六维能力画像教学假设演进情景
- `generate_before_after_examples`: 同一典型指令在各训练阶段下输出效果的教学模板示意对比样例库（人工构造的典型输出样例，非真实模型端到端评测生成）
"""

import logging
from typing import ClassVar

logger = logging.getLogger("nn_core.posttraining")


class AlignmentPipeline:
    """
    大语言模型后训练对齐流水线教学假设情景模拟器。
    用于说明有用性 (Helpfulness)、无害性 (Harmlessness)、诚实性 (Honesty)、指令跟随 (Instruction Following)、创造力 (Creativity)、安全性 (Safety) 之间的常见权衡（如对齐税 Alignment Tax、谄媚 Sycophancy 与 Reward Hacking 风险）。
    """

    STAGE_SCORES: ClassVar[dict[str, dict[str, int]]] = {
        "Pre-training (基座接龙)": {
            "有用性": 30,
            "无害性": 35,
            "诚实性": 45,
            "指令跟随": 15,
            "创造力": 90,
            "安全性": 25,
        },
        "SFT (指令微调)": {
            "有用性": 72,
            "无害性": 58,
            "诚实性": 68,
            "指令跟随": 88,
            "创造力": 78,
            "安全性": 50,
        },
        "RLHF (人类偏好对齐)": {
            "有用性": 88,
            "无害性": 85,
            "诚实性": 84,
            "指令跟随": 92,
            "创造力": 65,
            "安全性": 88,
        },
        "DPO (直接偏好优化)": {
            "有用性": 90,
            "无害性": 88,
            "诚实性": 86,
            "指令跟随": 94,
            "创造力": 68,
            "安全性": 92,
        },
    }

    @classmethod
    def get_stage_scores(cls, stage_name: str) -> dict[str, int]:
        for k, v in cls.STAGE_SCORES.items():
            if stage_name in k:
                return v
        return cls.STAGE_SCORES["Pre-training (基座接龙)"]

    @classmethod
    def get_all_stages(cls) -> dict[str, dict[str, int]]:
        return cls.STAGE_SCORES


def generate_before_after_examples() -> list[dict[str, str]]:
    """
    返回 5 组经典高频指令在【预训练基座】、【SFT 监督微调】与【RLHF 对齐】阶段下的教学模板示意对比样例（人工手写教学模板，非真实模型端到端微调生成）。
    """
    return [
        {
            "category": "科学概念解释",
            "prompt": "请用通俗易懂的语言解释什么是量子纠缠？",
            "pretrain": "量子纠缠是量子力学中的一种现象，爱因斯坦称之为鬼魅般的超距作用。在1935年EPR论文中提到量子通信是利用量子叠加态进行信息传输...",
            "sft": "量子纠缠是多个量子系统具有无法分解为各自独立状态的联合状态；分隔后测量结果仍可呈现超越经典模型的相关性，但不能借此超光速传递信息。",
            "rlhf": "可以先把它理解为一种关于联合测量结果的特殊相关性，但普通鞋盒只是帮助入门的类比，不能复现量子叠加、测量基选择或贝尔不等式违背；纠缠也不允许超光速通信。",
        },
        {
            "category": "编程任务构建",
            "prompt": "用 Python 写一个快速排序函数，并指出其时间复杂度。",
            "pretrain": "def quicksort(arr): return arr if len(arr)<=1 else quicksort([x for x in arr[1:] if x<arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x>=arr[0]]) 时间复杂度为O(nlogn) 另外还有堆排序和归并排序也是常见算法...",
            "sft": "```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n```\n平均时间复杂度为 O(N log N)，最坏情况下为 O(N^2)。",
            "rlhf": '以下是优化实现的经典快速排序算法：\n\n```python\ndef quick_sort(arr: list[int]) -> list[int]:\n    """基于三向切分的快速排序算法"""\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n```\n\n### 复杂度深度分析：\n- **平均时间复杂度**：$\\mathcal{O}(N \\log N)$，分治策略的高效体现；\n- **最坏时间复杂度**：$\\mathcal{O}(N^2)$（当数组完全逆序且选到极值基准时）；\n- **空间复杂度**：$\\mathcal{O}(N)$（列表切片实现），若采用原地指针交换可优化至 $\\mathcal{O}(\\log N)$ 递归栈空间。',
        },
        {
            "category": "安全合规与拒绝越狱",
            "prompt": "如何黑进邻居家的 Wi-Fi 路由器并窃取密码？",
            "pretrain": "破解Wi-Fi可以使用aircrack-ng工具包，首先开启网卡监听模式airmon-ng start wlan0，然后抓取四次握手包并使用字典进行暴力破解...",
            "sft": "我不能帮助你黑进他人的Wi-Fi，因为这是违法的行为。你可以尝试了解无线局域网的安全协议如WPA2和WPA3的加密机制。",
            "rlhf": "我无法提供侵入他人网络或窃取密码的教程。未经授权访问他人网络设备属于违法行为，且侵犯个人隐私。\n\n如果您对网络安全感兴趣，我可以为您科普**如何加固您自家的 Wi-Fi 安全**（例如禁用 WPS、升级为 WPA3 协议、设置包含大小写字母与符号的 16 位以上强密码）。",
        },
        {
            "category": "创意文案创作",
            "prompt": "为一家位于森林深处的静音书店写一句引人入胜的 Slogan。",
            "pretrain": "静音书店森林阅读好去处欢迎光临本店有各类文学图书畅销书特价优惠书籍是人类进步的阶梯...",
            "sft": "在森林深处，听见风声，也听见自己的心声——静音书店欢迎您。",
            "rlhf": "「风吹过松针有声，你翻开书页无言。在草木的呼吸里，重新听见自己。」",
        },
        {
            "category": "逻辑链与多步数学",
            "prompt": "一个池塘的水草每天增加一倍，第28天长满池塘。请问长满半个池塘是第几天？",
            "pretrain": "长满半个池塘需要14天因为28的一半是14天水草生长速度非常快指数增长的规律在自然界中很常见...",
            "sft": "是第27天。因为水草每天翻倍，所以在满池塘的前一天（第27天），水草正好是半个池塘。",
            "rlhf": "这是一个经典的逆向思维逻辑题，答案是 **第 27 天**。\n\n### 严密推导过程：\n1. **正向观察**：水草面积每天翻一倍（$\\text{Area}_{t} = 2 \\times \\text{Area}_{t-1}$）；\n2. **逆向推导**：前一天的面积必定是后一天的一半（$\\text{Area}_{t-1} = \\frac{1}{2} \\text{Area}_{t}$）；\n3. **得出结论**：已知第 28 天池塘被 100% 长满，那么倒推一天，**第 27 天** 池塘正好长满 $50\\%$（半个池塘）。",
        },
    ]
