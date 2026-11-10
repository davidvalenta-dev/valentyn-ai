"""Conservative training preflight and monitored process/Metal memory budgets."""

from dataclasses import dataclass

from .hardware import hardware_snapshot


@dataclass(frozen=True)
class ResourceBudget:
    max_rss_bytes: int = 3 * 1024 ** 3
    max_mps_driver_bytes: int = 3 * 1024 ** 3

    def preflight(self, parameters, batch, context, width, layers, heads):
        hardware = hardware_snapshot()
        estimate = (16 * parameters + batch * heads * context * context * layers * 16 +
                    batch * context * width * layers * 32)
        if estimate > self.max_rss_bytes:
            raise ValueError(f"Estimated training tensors ({estimate} bytes) exceed conservative memory budget")
        return {"hardware": hardware, "estimated_training_tensor_bytes": estimate,
                "memory_budget": {"process_rss_bytes": self.max_rss_bytes,
                                  "mps_driver_bytes": self.max_mps_driver_bytes},
                "note": "Estimate excludes runtime overhead. Monitoring checks process RSS and Metal driver counters separately."}

    def guard(self, device):
        from .neural import memory_measurement
        measured = memory_measurement(device)
        if measured["peak_process_rss_bytes"] > self.max_rss_bytes:
            raise RuntimeError("Process exceeded configured memory budget; refusing further training")
        if measured.get("mps_driver_bytes", 0) > self.max_mps_driver_bytes:
            raise RuntimeError("Metal driver exceeded configured memory budget; refusing further training")
