"""megatron_bench —— Megatron-LM 的纯 CPU 解析工具箱。

设计原则（见 deep-dive-megatron/RESEARCH_PLAN.md §1）：
本箱子里出来的所有数字都是 A 级证据——可重跑、可改参数、可被他人复算。
凡是需要真机才能定的数（吞吐、MFU 实测），不属于本箱子的职责范围。
"""

__version__ = "0.1.0"
