from typing import Any, Dict

import torch
import os
import sys
import subprocess
import json

from kernelgym.config import settings
from kernelgym.config.settings import KERNELBENCH_ROOT

ncu_path = settings.ncu_path
su_pwd = settings.su_pwd
ncu_metrics_path = settings.ncu_metrics_path
sys.path.insert(0, f"{ncu_path}/extras/python")
import ncu_report


def run_ncu_profiling(
    tempfile_path: str, kernel_exec_result: Dict[str, Any], device: torch.device = None
) -> Dict[str, Any]:
    if not os.path.exists(tempfile_path):
        kernel_exec_result.metadata["ncu_profiling"] = (
            f"Temp file {tempfile_path} does not exist after profiling."
        )
    else:
        with open(tempfile_path, "a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write('if __name__ == "__main__":\n')
            f.write("    model = ModelNew()\n")
            f.write("    inputs = get_inputs()\n")
            f.write("    outputs = model(*inputs)\n")

        try:
            device_idx = device.index if device.index is not None else 0
            rep_path = os.path.abspath("/tmp/tmp_ncu_report")
            cmd = f"echo {su_pwd} | sudo -S env PATH=$PATH {ncu_path}/ncu --set full --target-processes all -o {rep_path} -f {sys.executable} {tempfile_path} --device {device_idx}"
            result = subprocess.run(cmd, shell=True, capture_output=True)

            profiling_metrics = extract_ncu_profiling_metrics(f"{rep_path}.ncu-rep")

            kernel_exec_result.metadata["ncu_profiling"] = profiling_metrics

        except Exception as e:
            kernel_exec_result.metadata["ncu_profiling"] = {
                "ncu_profiling_error": str(e),
                "ncu_exit_code": str(result),
                "rep_path": f"{rep_path}.ncu-rep",
            }


def extract_ncu_profiling_metrics(report_path: str) -> Dict[str, Any]:
    json_path = KERNELBENCH_ROOT / settings.ncu_metrics_path
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            target_metrics = json.load(f)
    except Exception as e:
        return {"ncu_profiling_error": f"Failed to load metrics config json at {json_path}: {e}"}

    try:
        report = ncu_report.load_report(report_path)
    except Exception as e:
        return {"ncu_profiling_error": f"Failed to load ncu-rep: {e}"}

    for profile_range in report:
        for action in profile_range:
            kernel_name = action.name()
            profiling_metrics = {"kernel_name": kernel_name}
            
            try:
                for origin_name, config in target_metrics.items():
                    metric = action.metric_by_name(origin_name)
                    mapped_name = config.get("mapped_name", origin_name)
                    metric_type = config.get("type", "double")
                    
                    if metric_type == "string":
                        profiling_metrics[mapped_name] = metric.as_string()
                    elif metric_type == "uint64":
                        profiling_metrics[mapped_name] = metric.as_uint64()
                    else:
                        profiling_metrics[mapped_name] = metric.as_double()
                        
                return profiling_metrics
            except Exception as e:
                return {
                    "ncu_profiling_error": f"extracting metric failed for action {kernel_name}: {str(e)}"
                }
                
    return {"ncu_profiling_error": "No kernels found in report."}
