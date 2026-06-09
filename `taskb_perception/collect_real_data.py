"""
数据采集脚本 — 推荐从项目根目录运行根级 collect_real_data.py:
    cd ATEC2026_Simulation_Challenge
    python collect_real_data.py --num_images 500 --output datasets/real

本文件为兼容入口，逻辑与根目录版本一致。
"""

import os
import runpy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(_ROOT, "collect_real_data.py"), run_name="__main__")
