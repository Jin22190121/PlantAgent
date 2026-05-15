"""LangGraph AgentState definition."""
from typing import Any, Annotated, Literal, TypedDict
from operator import add


def _append_unique(left, right):
    """Reducer: append items from `right` to `left`, dropping duplicates.
    Order-preserving. Used so a step ID never appears twice in
    `completed_step_ids`."""
    seen = set(left or [])
    out = list(left or [])
    for item in (right or []):
        if not isinstance(item, str):
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _last_wins(_left, right):
    """Reducer: simply replace with the latest non-empty value."""
    return right if right else _left


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
    # Step progress — persisted across turns via the LangGraph checkpointer.
    # `completed_step_ids` accumulates uniquely; `current_step_id` is whatever
    # the planner last advised (the step the operator is currently working on).
    completed_step_ids: Annotated[list[str], _append_unique]
    current_step_id: Annotated[str, _last_wins]

    # Action loop
    proposed_action: ProposedAction
    approval_status: Literal["pending", "approved", "rejected", "modified", "n/a"]
    approval_reason: str
    last_tool_result: dict
    procedure_history: Annotated[list[dict], add]

    # Termination
    done: bool
