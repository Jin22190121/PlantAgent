import chromadb


# 절차서 샘플 데이터
PROCEDURES = [
    {
        "id": "E-0-1.1",
        "text": """절차서 E-0-1.1: 원자로 정지 확인
조건: 원자로 자동 정지 신호 발생 또는 수동 정지
행동: 모든 제어봉 완전 삽입 확인. 핵 계측기 지시치 감소 확인.
주의사항: 제어봉 미삽입 시 즉시 수동 삽입""",
        "category": "일반정지"
    },
    {
        "id": "E-1-1.1",
        "text": """절차서 E-1-1.1: ECCS 자동 기동 확인
조건: RCS 압력이 1700 psia 미만 감소 또는 SI 신호 발생
행동: SI 펌프 자동 기동 확인. 30초 이내 미확인 시 수동 기동.
주의사항: ECCS 절대 무단 차단 금지""",
        "category": "LOCA"
    },
    {
        "id": "E-1-1.2",
        "text": """절차서 E-1-1.2: 격납건물 격리 확인
조건: ECCS 기동 확인 후
행동: HI 신호 발생 확인. 격리 미완료 시 수동 격리.
주의사항: 격납건물 압력 설계 압력 초과 금지""",
        "category": "LOCA"
    },
    {
        "id": "E-1-2.1",
        "text": """절차서 E-1-2.1: RCS 압력 관리
조건: RCS 압력 1200 psia 미만
행동: 가압기 살수 밸브 닫힘 확인. 가압기 히터 차단.
주의사항: RCS 압력 급격 감소 시 파단 위치 파악 시도""",
        "category": "LOCA"
    },
    {
        "id": "ECA-0.0-1.1",
        "text": """절차서 ECA-0.0-1.1: 비상디젤발전기 수동 기동
조건: 소외전원 상실 및 EDG 자동 기동 실패
행동: EDG-1 또는 EDG-2 수동 기동. 전압 4160V, 주파수 60Hz 확인.
주의사항: 배터리 용량 8시간 한계. 비필수 부하 즉시 차단""",
        "category": "SBO"
    },
    {
        "id": "ECA-0.0-1.2",
        "text": """절차서 ECA-0.0-1.2: 증기구동 보조급수펌프 기동
조건: EDG 기동 실패 또는 소내 전원 전체 상실
행동: TDAFWP 자동 기동 확인. 미기동 시 수동 기동. SG 수위 확인.
주의사항: SG 수위 10% 미만 시 즉시 급수""",
        "category": "SBO"
    },
]


class ProcedureMCP:
    """
    절차서 RAG 검색 MCP 서버
    ChromaDB에 절차서를 임베딩하여 유사도 검색
    """
    SERVER_NAME = "procedure-mcp"

    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="procedures"
        )
        self._load_procedures()

    def _load_procedures(self):
        """절차서 데이터를 벡터DB에 로드"""
        if self.collection.count() > 0:
            return

        self.collection.add(
            documents=[p["text"] for p in PROCEDURES],
            ids=[p["id"] for p in PROCEDURES],
            metadatas=[{"category": p["category"]} for p in PROCEDURES]
        )
        print(f"✅ 절차서 {self.collection.count()}개 로드 완료")

    def list_tools(self):
        return [
            {
                "name": "search_procedure",
                "description": "상황 설명으로 관련 절차서 RAG 검색"
            },
            {
                "name": "get_caution_notes",
                "description": "특정 절차서 ID의 주의사항 반환"
            }
        ]

    def call_tool(self, tool_name: str, args: dict):
        if tool_name == "search_procedure":
            return self._search_procedure(
                args.get("situation", ""),
                args.get("n_results", 2)
            )
        elif tool_name == "get_caution_notes":
            return self._get_caution_notes(
                args.get("procedure_id", "")
            )
        return {"error": f"알 수 없는 도구: {tool_name}"}

    def _search_procedure(self, situation: str, n_results: int):
        results = self.collection.query(
            query_texts=[situation],
            n_results=n_results
        )
        procedures = []
        for i, doc in enumerate(results["documents"][0]):
            procedures.append({
                "id": results["ids"][0][i],
                "content": doc
            })
        return {
            "query": situation,
            "found": len(procedures),
            "procedures": procedures
        }

    def _get_caution_notes(self, procedure_id: str):
        results = self.collection.query(
            query_texts=[procedure_id],
            n_results=1
        )
        if results["documents"][0]:
            content = results["documents"][0][0]
            lines = content.split("\n")
            cautions = [l for l in lines if "주의" in l]
            return {
                "procedure_id": procedure_id,
                "cautions": cautions
            }
        return {"error": "절차서를 찾을 수 없습니다."}