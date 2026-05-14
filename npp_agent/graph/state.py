"""LangGraph AgentState definition."""
from typing import Any, Annotated, Literal, TypedDict
from operator import add


class ProposedAction(TypedDict, total=False):
    tool: str
    args: dict[str, Any]
    rationale: str
    cited_step_id: str
    expected_outcome: str
    cautions: list[str]


class AgentState(TypedDict, total=False):
    # I/O
    operator_input: str
    final_answer: str
    # Accumulated user-visible messages this turn — `add` reducer so multiple
    # nodes can append (plan_action, verify_outcome, respond …) and the
    # respond node consolidates them into a single `final_answer`.
    final_messages: Annotated[list[str], add]

    # Plant context
    plant_state: dict
    alarms: list[dict]

    # Procedure context
    retrieved_procedures: list[dict]
    current_step_id: str

    # Action loop
    proposed_action: ProposedAction
    approval_status: Literal["pending", "approved", "rejected", "modified", "n/a"]
    approval_reason: str
    last_tool_result: dict
    procedure_history: Annotated[list[dict], add]

    # Termination
    done: bool
