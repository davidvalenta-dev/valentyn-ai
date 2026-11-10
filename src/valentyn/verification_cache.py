"""Experiment-local reuse of identical source previously accepted by Lean.

No persistent files are trusted. Every distinct source is compiled at least once
in this process; returned cache hits are explicit. The ordinary verifier remains
unchanged and archive audits always rerun Lean independently.
"""
from dataclasses import replace
import hashlib
import os
from pathlib import Path

from .verifier import LeanVerifier, lean_binary, render


class VerificationCache:
    def __init__(self, project, max_entries=10000):
        self.verifier = LeanVerifier(project)
        self.cache = {}
        self.max_entries = max_entries
        self.hits = self.compilations = 0

    def _identity(self):
        binary = lean_binary()
        if binary is None:
            return None
        path = Path(binary).resolve()
        info = path.stat()
        toolchain = self.verifier.project / 'lean-toolchain'
        pin = toolchain.read_bytes() if toolchain.exists() else b''
        return str(path), info.st_mtime_ns, info.st_size, pin, os.environ.get('ELAN_TOOLCHAIN')

    def verify(self, goal, proof):
        try:
            source = render(goal, proof)
            identity = self._identity()
        except (ValueError, KeyError, RecursionError, OSError):
            return self.verifier.verify(goal, proof)
        key = (identity, hashlib.sha256(source.encode()).hexdigest())
        if identity is not None and key in self.cache:
            self.hits += 1
            return replace(self.cache[key], reason='verified_in_process_cache', elapsed=0)
        result = self.verifier.verify(goal, proof)
        self.compilations += 1
        if result.accepted and len(self.cache) < self.max_entries:
            self.cache[key] = result
        return result
