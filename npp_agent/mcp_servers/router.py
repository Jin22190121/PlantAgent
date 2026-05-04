class MCPRouter:
    """
    MCPRouter: 도구 이름으로 서버를 자동 라우팅
    Agent는 도구 이름만 알면 어느 서버인지 몰라도 됨
    """

    def __init__(self, servers: list):
        self.servers = servers
        self.tool_map = {}

        for server in servers:
            for tool in server.list_tools():
                self.tool_map[tool["name"]] = server

        print(f"✅ MCPRouter 초기화: {len(self.tool_map)}개 도구 등록")
        for name, server in self.tool_map.items():
            print(f"   - {name} → [{server.SERVER_NAME}]")

    def call(self, tool_name: str, args: dict):
        server = self.tool_map.get(tool_name)
        if not server:
            return {"error": f"도구를 찾을 수 없음: {tool_name}"}

        print(f"\n  🔧 [{server.SERVER_NAME}] {tool_name}")
        print(f"  📥 입력: {args}")

        result = server.call_tool(tool_name, args)
        print(f"  📤 완료")
        return result