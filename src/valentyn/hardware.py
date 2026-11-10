"""Lightweight hardware detection without loading an optional neural runtime."""

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys


def hardware_snapshot() -> dict:
    memory = None
    available = None
    chip = platform.processor()
    try:
        if platform.system() == "Darwin":
            memory = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                                                 text=True, timeout=5).strip())
            chip = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True, timeout=5).strip()
            statistics = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=5)
            page_size = int(re.search(r"page size of ([0-9]+) bytes", statistics)[1])
            pages = {name: int(value) for name, value in re.findall(r"^(Pages [^:]+):\s+([0-9]+)", statistics, re.M)}
            available = page_size * sum(pages.get(name, 0) for name in
                                       ("Pages free", "Pages inactive", "Pages speculative"))
        else:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {"machine": platform.machine(), "chip": chip, "system": platform.system(),
            "os_version": platform.platform(), "python": sys.version.split()[0],
            "python_executable": sys.executable, "available_memory_estimate_bytes": available,
            "free_disk_bytes": shutil.disk_usage(os.getcwd()).free,
            "available_memory_note": "macOS estimate includes reclaimable inactive/speculative pages; not a guaranteed allocation.",
            "physical_memory_bytes": memory, "conservative_target_bytes": 8 * 1024 ** 3,
            "torch_installed": importlib.util.find_spec("torch") is not None,
            "mlx_installed": importlib.util.find_spec("mlx") is not None}
