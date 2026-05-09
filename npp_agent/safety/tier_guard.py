"""R/A/E tier classification + approval token verification.

Each MCP tool is classified into one of three safety tiers:

  R (Read)     — read-only, auto-execute (e.g. get_plant_state, search_*)
  A (Advise)   — produces advisory output, no plant change (e.g. log_action)
  E (Execute)  — mutates the simulator/plant (e.g. set_pzr_heater, start_rcp)

E-tier tools MUST go through HITL approval. The router checks for an
ApprovalToken issued by the approval_gate node before executing.
"""
from __future__ import annotations
import hmac
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    R = "R"
    A = "A"
    E = "E"


# Tool → tier classification
TIER_MAP: dict[str, Tier] = {
    # READ
    "get_plant_state":            Tier.R,
    "get_alarm_list":             Tier.R,
    "search_procedure":           Tier.R,
    "get_step":                   Tier.R,
    "get_next_step":              Tier.R,
    "search_by_mode_transition":  Tier.R,
    "get_cautions":               Tier.R,
    "get_step_by_action":         Tier.R,
    "get_session_summary":        Tier.R,
    # ADVISE (produce records, no plant change)
    "log_action":                 Tier.A,
    # EXECUTE (mutate simulator state)
    "set_pzr_heater":             Tier.E,
    "set_pzr_spray":              Tier.E,
    "set_rhr_pump":               Tier.E,
    "start_rcp":                  Tier.E,
    "stop_rcp":                   Tier.E,
    "set_charging_flow":          Tier.E,
    "set_letdown_flow":           Tier.E,
    "set_sg_level_target":        Tier.E,
    "open_msiv":                  Tier.E,
    "advance_time":               Tier.E,
    "reset_simulator":            Tier.E,
}


def classify(tool_name: str) -> Tier:
    """Default to E when unknown — fail safe."""
    return TIER_MAP.get(tool_name, Tier.E)


def requires_approval(tool_name: str) -> bool:
    return classify(tool_name) == Tier.E


# ── Approval token (HMAC-signed) ─────────────────────────
_SECRET_ENV = "NPP_APPROVAL_SECRET"

def _secret() -> bytes:
    s = os.environ.get(_SECRET_ENV)
    if not s:
        # Generate ephemeral process-local secret if user didn't set one.
        # This means tokens are valid only within this server instance.
        global _ephemeral_secret
        try:
            _ephemeral_secret
        except NameError:
            _ephemeral_secret = os.urandom(32).hex()
        s = _ephemeral_secret
    return s.encode()


@dataclass(frozen=True)
class ToolApprovalToken:
    tool: str
    args_hash: str
    issued_at: int
    signature: str
    thread_id: str

    def serialize(self) -> dict:
        return {
            "tool": self.tool,
            "args_hash": self.args_hash,
            "issued_at": self.issued_at,
            "signature": self.signature,
            "thread_id": self.thread_id,
        }


def _hash_args(args: dict) -> str:
    import json, hashlib
    blob = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def issue_token(tool: str, args: dict, thread_id: str) -> ToolApprovalToken:
    args_hash = _hash_args(args)
    issued_at = int(time.time())
    payload = f"{tool}|{args_hash}|{issued_at}|{thread_id}".encode()
    sig = hmac.new(_secret(), payload, "sha256").hexdigest()
    return ToolApprovalToken(tool, args_hash, issued_at, sig, thread_id)


def verify_token(token: Optional[ToolApprovalToken],
                 tool: str, args: dict, thread_id: str,
                 max_age_s: int = 600) -> bool:
    """True iff token is well-formed, current and matches (tool, args, thread)."""
    if token is None:
        return False
    if token.tool != tool:
        return False
    if token.args_hash != _hash_args(args):
        return False
    if token.thread_id != thread_id:
        return False
    if int(time.time()) - token.issued_at > max_age_s:
        return False
    payload = f"{token.tool}|{token.args_hash}|{token.issued_at}|{token.thread_id}".encode()
    expected = hmac.new(_secret(), payload, "sha256").hexdigest()
    return hmac.compare_digest(expected, token.signature)
