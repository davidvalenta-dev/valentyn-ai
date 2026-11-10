"""Resource-limited, isolated-interpreter strategy evaluation worker.

Only validated numeric policy data is mutable. This is NOT an OS security sandbox
for arbitrary Python code; the system never executes candidate-authored code.
"""

from dataclasses import asdict
import json
import resource
import sys

from .logic import parse
from .policy import Policy
from .search import Prover


def main() -> int:
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    cap = min(128, hard) if hard != resource.RLIM_INFINITY else 128
    resource.setrlimit(resource.RLIMIT_NOFILE, (cap, cap))
    request = json.loads(sys.stdin.read(65537))
    if set(request) != {"policy", "tasks", "nodes", "depth", "timeout"}:
        raise ValueError("Invalid worker request")
    policy = Policy.from_dict(request["policy"])
    if not isinstance(request["tasks"], list) or len(request["tasks"]) > 100:
        raise ValueError("Too many tasks")
    prover = Prover(max_nodes=request["nodes"], max_depth=request["depth"], timeout=request["timeout"])
    results = []
    for task in request["tasks"]:
        goal = parse(task["formula"])
        results.append({"task_id": task["id"], "formula": task["formula"],
                        **asdict(prover.solve(goal, policy.choose(goal)))})
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
