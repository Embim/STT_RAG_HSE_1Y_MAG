import os
os.environ.setdefault("LLM_MODEL", "test/model")
os.environ.setdefault("LLM_API_KEY_1", "x")
os.environ.setdefault("LLM_API_KEY_2", "x")
os.environ.setdefault("LLM_API_KEY_3", "x")

import numpy as np
import pytest


# ── фейковый Weaviate-менеджер ─────────────────────────────────────────
class _FakeAggregate:
    def __init__(self, count):
        self._count = count

    def over_all(self, total_count=True):
        class _R:
            pass
        r = _R()
        r.total_count = self._count
        return r


class _FakeCollection:
    def __init__(self, count, near_ids=None):
        self.aggregate = _FakeAggregate(count)
        self._near_ids = near_ids or []

    class _Obj:
        def __init__(self, uuid):
            self.uuid = uuid

    @property
    def query(self):
        coll = self

        class _Q:
            def near_text(self, query, limit):
                class _Resp:
                    pass
                r = _Resp()
                r.objects = [_FakeCollection._Obj(i) for i in coll._near_ids[:limit]]
                return r
        return _Q()


class _FakeManager:
    def __init__(self, count, near_ids=None):
        self.collection = _FakeCollection(count, near_ids)


def _synthetic(n_per=20, dim=40, n_clusters=3, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim) * 5
    vecs, metas, ids = [], [], []
    texts = [
        "нейронные сети глубокое обучение слой attention",
        "рекомендательные системы коллаборативная фильтрация эмбеддинг",
        "градиентный бустинг деревья решений признаки",
    ]
    for c in range(n_clusters):
        for i in range(n_per):
            vecs.append(centers[c] + rng.randn(dim))
            metas.append({
                "title": f"Лекция {c}", "source_url": "http://x", "start": i * 10.0,
                "hash": f"h{c}_{i}", "chunk_index": i,
                "snippet": texts[c], "text": texts[c] + f" пример {i}",
            })
            ids.append(f"id-{c}-{i}")
    return ids, np.asarray(vecs, dtype=np.float32), metas


@pytest.fixture
def patched(monkeypatch):
    """Подменяет fetch + менеджер, сбрасывает кэш."""
    import system.embedding_map as em
    import system.llm.llm_services as svc

    ids, vecs, metas = _synthetic()
    em.reset_cache()
    monkeypatch.setattr(em, "_fetch_vectors", lambda manager, limit: (ids, vecs, metas))
    mgr = _FakeManager(count=len(ids), near_ids=["id-0-0", "id-0-1", "id-1-0"])
    monkeypatch.setattr(svc, "get_chat_vectore_store_manager", lambda: mgr)
    return em, mgr, ids


def test_build_map_structure(patched):
    em, _, ids = patched
    res = em.build_map(force=True)
    assert res["count"] == len(ids)
    assert res["reducer"] in ("umap", "pca")
    assert len(res["points"]) == len(ids)
    p = res["points"][0]
    assert set(p) >= {"id", "x", "y", "z", "cluster", "title", "source_url", "start", "snippet"}
    # координаты нормированы в ~[-50, 50]
    assert all(abs(pt["x"]) <= 50.01 and abs(pt["y"]) <= 50.01 and abs(pt["z"]) <= 50.01
               for pt in res["points"])
    assert res["topics"], "должны быть темы"
    for t in res["topics"]:
        assert set(t) >= {"cluster", "label", "color", "size"}
        assert t["color"].startswith("#") and len(t["color"]) == 7
    # сумма размеров тем = число точек
    assert sum(t["size"] for t in res["topics"]) == len(ids)


def test_cache_reuse_and_invalidation(patched, monkeypatch):
    em, _, _ = patched
    calls = {"n": 0}
    orig = em._reduce

    def counting(vectors, kind):
        calls["n"] += 1
        return orig(vectors, kind)
    monkeypatch.setattr(em, "_reduce", counting)

    em.build_map(force=True)
    assert calls["n"] == 1
    em.build_map()            # тот же count → из кэша, без пересчёта
    assert calls["n"] == 1
    em.build_map(force=True)  # форс → пересчёт
    assert calls["n"] == 2


def test_empty_corpus(monkeypatch):
    import system.embedding_map as em
    import system.llm.llm_services as svc
    em.reset_cache()
    monkeypatch.setattr(em, "_fetch_vectors",
                        lambda manager, limit: ([], np.empty((0, 0), dtype=np.float32), []))
    monkeypatch.setattr(svc, "get_chat_vectore_store_manager", lambda: _FakeManager(count=0))
    res = em.build_map(force=True)
    assert res["points"] == [] and res["topics"] == []


def test_locate_highlights_and_centroid_marker(patched):
    em, _, _ = patched
    res_map = em.build_map(force=True)
    res = em.locate("что такое коллаборативная фильтрация", top_k=3)
    assert res["highlight_ids"] == ["id-0-0", "id-0-1", "id-1-0"]
    # маркер = центроид координат подсвеченных точек (внутри облака)
    assert res["marker"] is not None
    pts = {p["id"]: p for p in res_map["points"]}
    exp_x = sum(pts[i]["x"] for i in res["highlight_ids"]) / 3
    assert abs(res["marker"]["x"] - exp_x) < 0.01


def test_locate_marker_none_when_neighbors_absent(monkeypatch):
    """Если near_text вернул id, которых нет в карте — маркер None, подсветка есть."""
    import system.embedding_map as em
    import system.llm.llm_services as svc
    ids, vecs, metas = _synthetic()
    em.reset_cache()
    monkeypatch.setattr(em, "_fetch_vectors", lambda manager, limit: (ids, vecs, metas))
    # near_text возвращает id, которых нет среди точек карты
    mgr = _FakeManager(count=len(ids), near_ids=["ghost-1", "ghost-2"])
    monkeypatch.setattr(svc, "get_chat_vectore_store_manager", lambda: mgr)
    em.build_map(force=True)
    res = em.locate("запрос", top_k=2)
    assert res["highlight_ids"] == ["ghost-1", "ghost-2"]  # подсветка возвращается
    assert res["marker"] is None                            # но центроид построить не из чего


def test_hsl_to_hex_and_palette():
    import system.embedding_map as em
    assert em._hsl_to_hex(0, 1.0, 0.5) == "#ff0000"
    pal = em._palette(6)
    assert len(pal) == 6 and all(c.startswith("#") and len(c) == 7 for c in pal)
