"""主流程可用的轻量本地规程检索；向量检索模块保持独立。"""

import json
from pathlib import Path

from src.schemas.state import (
    DiagnosisState, EvidenceKind, EvidenceRef, ManualEvidence, ManualState,
    ProcedureRequirement, RecommendedAction,
)


def retrieve_local_manual(state: DiagnosisState) -> ManualState:
    """按实际规则码/关键词检索 JSON 条款，不加载模型、不伪造向量分数。

    JSON 的 clauses 每条包含 doc_id/chunk_id/section/quote，以及匹配用的
    rule_codes/keywords 和可选 requirement/action/mechanism。
    score 表示匹配词覆盖率，channel=KEYWORD；它不是语义相似度。
    """
    path = Path(state.context.manual_uri).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    flags = set(state.data.threshold_flags)
    query = state.engineer_text.text + " " + (state.context.alarm_code or "")
    matches = []
    for clause in payload["clauses"]:
        terms = clause.get("rule_codes", []) + clause.get("keywords", [])
        hits = sum(term in flags or term in query for term in terms)
        if hits:
            matches.append((hits / len(terms), clause))
    matches.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
    evidences, requirements, actions, mechanisms = [], [], [], []
    for score, clause in matches[:5]:
        ref = EvidenceRef(kind=EvidenceKind.MANUAL,
                          id=f"{clause['doc_id']}#{clause['chunk_id']}",
                          locator=f"{path} · {clause['section']}")
        evidences.append(ManualEvidence(
            doc_id=clause["doc_id"], chunk_id=clause["chunk_id"],
            section=clause["section"], quote=clause["quote"],
            page=clause.get("page", 0), score=score, channel="KEYWORD"))
        if clause.get("requirement"):
            requirements.append(ProcedureRequirement(requirement=clause["requirement"], source_ref=ref))
        if clause.get("action"):
            actions.append(RecommendedAction(action=clause["action"], priority="HIGH", source_ref=ref))
        if clause.get("mechanism"):
            mechanisms.append(clause["mechanism"])
    return ManualState(status="HIT" if evidences else "NO_HIT", evidences=evidences,
                       requirements=requirements, actions=actions, mechanism="；".join(mechanisms))
