"""第5回 発展: Graphの候補を質問との近さ順に選ぶ（rerank）。固定ベクトルで確認し、APIは呼ばない。"""

import json

import pytest

from app import retrieval
from scripts.build_rag_index import build_index

VECTORS = {"p001": [1, 0], "p002": [0, 1], "p003": [0.6, 0.8], "p004": [0.9, 0.436]}


@pytest.fixture
def graph_env(tmp_path, monkeypatch):
    """p001が入口。p002は前後ページ（遠い）、p004は共通用語でつながる（近い）。"""
    pages = [
        {"id": f"lec01-main-{pid}", "title": f"ページ{pid}", "page": n, "content": f"本文{pid}"}
        for n, pid in enumerate(VECTORS, 1)
    ]
    source = tmp_path / "lectures.json"
    source.write_text(json.dumps([{"id": "lec01-main", "title": "講義1", "content": [
        {"section": "章", "content": pages}]}], ensure_ascii=False), encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    build_index(source, database, embedder=lambda text, task: next(v for k, v in VECTORS.items() if k in text) if task == "RETRIEVAL_DOCUMENT" else [1, 0])

    def node(i, kind="page"):
        return {"id": i, "kind": kind, "title": i}
    ids = [f"lec01-main-{pid}" for pid in VECTORS]
    graph = {
        "nodes": [node(i) for i in ids] + [node("term:t", "term")],
        "edges": [{"source": ids[0], "target": ids[1], "relation": "next_page"},
                  {"source": ids[0], "target": "term:t", "relation": "mentions"},
                  {"source": ids[3], "target": "term:t", "relation": "mentions"}],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "RAG_GRAPH", graph_path)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])
    monkeypatch.setattr(retrieval, "load_pages", lambda *_: [
        {"page_id": p["id"], "document_id": "lec01-main", "lecture": 1, "document_title": "講義1",
         "section": "章", "title": p["title"], "page": p["page"], "content": p["content"]} for p in pages])
    monkeypatch.setattr(retrieval, "GRAPH_START_K", 1)
    monkeypatch.setattr(retrieval, "GRAPH_MAX_RESULTS", 2)  # 入口1件 + 候補1件
    return ids


def test_default_prefers_fewer_hops(graph_env):
    result = retrieval.search_graph("質問")
    assert [p["page_id"] for p in result["expanded"]] == [graph_env[1]]  # 前後ページ（1 hop）が先


def test_rerank_prefers_pages_close_to_the_question(graph_env):
    result = retrieval.search_graph("質問", rerank=True)
    assert [p["page_id"] for p in result["seeds"]] == [graph_env[0]]
    assert [p["page_id"] for p in result["expanded"]] == [graph_env[3]]  # 遠い前後ページより、近い用語つながりのページ
    assert result["expanded"][0]["distance"] < 0.2
