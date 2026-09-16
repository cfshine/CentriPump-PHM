"""pytest 引导。

把项目根加入 sys.path，使以下两种导入都成立：
    from src.schemas.state import DiagnosisState
    from rules.thresholds import TEMP_TRIP_C

这样不需要打包（pyproject/install）也能直接跑测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
