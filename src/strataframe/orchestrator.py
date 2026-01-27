# src/strataframe/orchestrator.py
from __future__ import annotations

from strataframe.steps.step0_index_gr import Step0Config, Step0Paths, run_step0_index_gr
from strataframe.steps.step1_build_bins import Step1Config, Step1Paths, run_step1_build_bins

__all__ = [
    "Step0Config",
    "Step0Paths",
    "run_step0_index_gr",
    "Step1Config",
    "Step1Paths",
    "run_step1_build_bins",
]

# Step 2/3 are optional at import time; they may depend on extras or in-progress modules.
try:  # pragma: no cover
    from strataframe.pipelines.step2_reps import Step2RepsConfig as Step2Config  # type: ignore
    from strataframe.steps.step2_run import run_step2  # type: ignore

    __all__ += ["Step2Config", "run_step2"]
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover
    from strataframe.steps.step3_run_correlation import Step3Config, Step3Paths, run_step3  # type: ignore

    __all__ += ["Step3Config", "Step3Paths", "run_step3"]
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover
    from strataframe.steps.step3_run_correlation import Step3Config, Step3Paths, run_step3  # type: ignore

    __all__ += ["Step3Config", "Step3Paths", "run_step3"]
except Exception:  # pragma: no cover
    pass
