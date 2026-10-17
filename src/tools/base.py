"""base class for all tools

- extract(query) -> argument or None   pull what the tool needs out of the user's text
- run(argument)  -> ToolResult         do the work
- name/description                     for logs, docs and the registry

tools dont raise exceptions: executor decides what to do. 
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    tool: str
    ok: bool
    output: str = ""                              
    data: dict = field(default_factory=dict)      
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    extract: Callable[[str], str | None]
    run_fn: Callable[[str], ToolResult]

    def run(self, argument: str) -> ToolResult:
        #run the tool, time it, turn any unexpected crash into a failed tool result.
        start = time.perf_counter()
        try:
            result = self.run_fn(argument)
        except Exception as e:  
            result = ToolResult(self.name, ok=False, error=f"{type(e).__name__}: {e}")
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result
