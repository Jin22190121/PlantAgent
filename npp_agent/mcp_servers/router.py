"""MCP router with R/A/E tier safety gating.

Read tools execute freely. Execute-tier tools require an approval token
issued by the LangGraph approval_gate node. Direct UI button calls (where
the operator presses a button themselves) bypass the token check via
`source='operator_ui'`.
"""
from npp_agent.safety import classify, requires_approval, verify_token, Tier


class MCPRouter:
    def __init__(self, servers: list):
        self.servers = servers
        self.tool_map = {}
        for server in servers:
            for tool in server.list_tools():
                self.tool_map[tool["name"]] = server

        print(f"✅ MCPRouter 초기화: {len(self.tool_map)}개 도구")
        for name, server in self.tool_map.items():
            t = classify(name)
            print(f"   - [{t.value}] {name} → [{server.SERVER_NAME}]")

    def call(self, tool_name: str, args: dict, *,
             source: str = "agent",
             approval_token=None,
             thread_id: str = "") -> dict:
        """
        source: 'agent' | 'operator_ui' | 'graph_internal'
          - agent: LLM tool call. E-tier requires approval_token.
          - operator_ui: button press by human; bypasses tier gate.
          - graph_internal: invoked from a verified graph node post-approval.
        """
        server = self.tool_map.get(tool_name)
        if not server:
            return {"error": f"도구를 찾을 수 없음: {tool_name}"}

        # ── tier guard ──
        if requires_approval(tool_name) and source == "agent":
            if not verify_token(approval_token, tool_name, args, thread_id):
                return {
                    "error": "TIER-E action requires operator approval",
                    "tool": tool_name,
                    "tier": "E",
                    "needs_approval": True,
                }

        print(f"  🔧 [{server.SERVER_NAME}] {tool_name}({args}) [{classify(tool_name).value}/{source}]")
        return server.call_tool(tool_name, args)

    # Convenience for graph nodes
    def call_internal(self, tool_name: str, args: dict) -> dict:
        return self.call(tool_name, args, source="graph_internal")
