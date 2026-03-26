"""
Unified result schema shared by all sub-modules.
Every module's run.py must output a JSON object matching this structure.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class UnifiedResult(BaseModel):
    module_id: str
    module_name: str
    status: str          # "success" | "error" | "disabled"
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    summary: str
    error: Optional[str] = None

    def to_template_dict(self) -> dict[str, Any]:
        """Flat dict convenient for Jinja2 rendering."""
        return {
            "id": self.module_id,
            "name": self.module_name,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "summary": self.summary,
            "error": self.error,
            "ok": self.status == "success",
        }
