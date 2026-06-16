"""3D-карта эмбеддингового пространства («облако тем»).

Тянет векторы чанков из Weaviate, снижает размерность до 3D, кластеризует в
авто-темы и подписывает их. Результат кэшируется (вместе с обученным
редьюсером — чтобы можно было спроецировать новый запрос в то же пространство).

Тяжёлые зависимости (scikit-learn, umap-learn) импортируются ЛЕНИВО внутри
функций: модуль грузится и без них (на дев-машине), а конкретный вызов
build_map/locate уже потребует scikit-learn (umap опционален → fallback на PCA).

Точка = чанк. Цвет = авто-тема (кластер KMeans), подпись темы = топ-термины
(c-TF-IDF по кластерам). У точки есть title/source_url/start — клик на фронте
открывает лекцию на нужном моменте.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from settings import settings

logger = logging.getLogger(__name__)

# Компактный список русских стоп-слов + общая «болтовня» лекций, чтобы подписи
# тем были содержательными, а не «вот», «это», «значит».
_RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
    "всегда", "конечно", "всю", "между", "это", "вообще", "просто", "значит",
    "типа", "какие", "вроде", "именно", "очень", "также", "которые", "который",
}


@dataclass
class _MapCache:
    points: List[dict]
    topics: List[dict]
    count: int          # число объектов в коллекции на момент сборки (для инвалидации)
    dims: int           # исходная размерность эмбеддинга
    reducer_kind: str   # "umap" | "pca" | "none"


_cache: Optional[_MapCache] = None
_lock = threading.Lock()


# ── вычислительные шаги ────────────────────────────────────────────────
def _fetch_vectors(manager, limit: int) -> Tuple[List[str], np.ndarray, List[dict]]:
    """Достать из Weaviate id + вектор + метаданные каждого чанка."""
    resp = manager.collection.query.fetch_objects(
        limit=limit,
        include_vector=True,
        return_properties=["text", "metadata"],
    )
    ids: List[str] = []
    vectors: List[List[float]] = []
    metas: List[dict] = []
    for obj in resp.objects:
        vec = obj.vector
        if isinstance(vec, dict):  # named vectors → берём дефолтный
            vec = vec.get("default") or next(iter(vec.values()), None)
        if not vec:
            continue
        meta = manager._safe_parse_metadata(obj.properties.get("metadata", "{}"))
        text = str(obj.properties.get("text", ""))
        ids.append(str(obj.uuid))
        vectors.append(list(vec))
        metas.append({
            "title": meta.get("title") or meta.get("source_file_name") or "—",
            "source_url": meta.get("source_url"),
            "start": meta.get("start_sec"),
            "hash": meta.get("hash"),
            "chunk_index": meta.get("chunk_index"),
            "snippet": text[:240],
            "text": text,
        })
    if not vectors:
        return ids, np.empty((0, 0), dtype=np.float32), metas
    return ids, np.asarray(vectors, dtype=np.float32), metas


def _reduce(vectors: np.ndarray, kind: str) -> Tuple[np.ndarray, Any, str]:
    """Снизить размерность до 3D. Возвращает (coords[n,3], обученный редьюсер, вид)."""
    want_umap = kind in ("auto", "umap")
    if want_umap:
        try:
            import umap  # type: ignore
            n = vectors.shape[0]
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(15, max(2, n - 1)),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            coords = reducer.fit_transform(vectors)
            return np.asarray(coords, dtype=np.float32), reducer, "umap"
        except ImportError:
            if kind == "umap":
                raise RuntimeError("EMBEDDING_MAP_REDUCER=umap, но umap-learn не установлен")
            logger.info("umap-learn не найден — fallback на PCA")
        except Exception as e:  # noqa: BLE001 — на маленьких/вырожденных данных UMAP может падать
            logger.warning("UMAP не справился (%s) — fallback на PCA", e)
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=min(3, vectors.shape[1], max(1, vectors.shape[0])))
    coords = reducer.fit_transform(vectors)
    if coords.shape[1] < 3:  # добиваем нулями до 3D, если измерений мало
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    return np.asarray(coords, dtype=np.float32), reducer, "pca"


def _normalize(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Центрировать в 0 и отмасштабировать в ~[-50, 50] для Three.js."""
    center = coords.mean(axis=0)
    centered = coords - center
    scale = float(np.abs(centered).max()) or 1.0
    norm = centered / scale * 50.0
    return norm.astype(np.float32), center.astype(np.float32), scale


def _cluster(vectors: np.ndarray) -> Tuple[np.ndarray, int]:
    """KMeans с эвристическим k. Возвращает (labels[n], k)."""
    from sklearn.cluster import KMeans
    n = vectors.shape[0]
    k = max(2, min(24, round(math.sqrt(n / 2)))) if n >= 4 else 1
    k = min(k, n)
    if k <= 1:
        return np.zeros(n, dtype=int), 1
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
    return labels.astype(int), k


def _label_clusters(texts: List[str], labels: np.ndarray, k: int) -> dict:
    """Подписать каждый кластер топ-терминами (c-TF-IDF по кластерам)."""
    if k <= 1:
        return {0: "все темы"}
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = []
    for c in range(k):
        docs.append(" ".join(t for t, l in zip(texts, labels) if l == c))
    try:
        vec = TfidfVectorizer(
            max_features=4000,
            token_pattern=r"(?u)\b[а-яёa-z][а-яёa-z-]{2,}\b",
            stop_words=list(_RU_STOPWORDS),
            lowercase=True,
        )
        m = vec.fit_transform(docs)
        terms = vec.get_feature_names_out()
    except ValueError:
        return {c: f"тема {c + 1}" for c in range(k)}
    out = {}
    for c in range(k):
        row = m[c].toarray()[0]
        top_idx = row.argsort()[::-1][:3]
        words = [terms[i] for i in top_idx if row[i] > 0]
        out[c] = ", ".join(words) if words else f"тема {c + 1}"
    return out


def _palette(k: int) -> List[str]:
    """k визуально различимых цветов (равномерно по тону HSL)."""
    colors = []
    for i in range(k):
        h = (i * 360.0 / max(1, k)) % 360
        colors.append(_hsl_to_hex(h, 0.62, 0.6))
    return colors


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = l - c / 2
    if h < 60:   r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return "#{:02x}{:02x}{:02x}".format(
        round((r + m) * 255), round((g + m) * 255), round((b + m) * 255)
    )


def _serialize(cache: _MapCache) -> dict:
    return {
        "points": cache.points,
        "topics": cache.topics,
        "count": cache.count,
        "dims": cache.dims,
        "reducer": cache.reducer_kind,
    }


# ── публичный API ──────────────────────────────────────────────────────
def build_map(force: bool = False) -> dict:
    """Собрать (или вернуть из кэша) 3D-карту эмбеддингов.

    Кэш инвалидируется, когда меняется число объектов в коллекции или force=True.
    """
    global _cache
    from system.llm.llm_services import get_chat_vectore_store_manager
    manager = get_chat_vectore_store_manager()
    try:
        count = manager.collection.aggregate.over_all(total_count=True).total_count
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"vector db недоступна: {e}") from e

    with _lock:
        if _cache is not None and not force and _cache.count == count:
            return _serialize(_cache)

        ids, vectors, metas = _fetch_vectors(manager, settings.EMBEDDING_MAP_MAX_POINTS)
        n = vectors.shape[0]
        if n == 0:
            empty = _MapCache([], [], count, 0, "none")
            _cache = empty
            return _serialize(empty)

        coords3d, _reducer, kind = _reduce(vectors, settings.EMBEDDING_MAP_REDUCER)
        norm, _center, _scale = _normalize(coords3d)
        labels, k = _cluster(vectors)
        label_map = _label_clusters([m["text"] for m in metas], labels, k)
        palette = _palette(k)

        points = []
        for i in range(n):
            c = int(labels[i])
            points.append({
                "id": ids[i],
                "x": round(float(norm[i, 0]), 3),
                "y": round(float(norm[i, 1]), 3),
                "z": round(float(norm[i, 2]), 3),
                "cluster": c,
                "title": metas[i]["title"],
                "source_url": metas[i]["source_url"],
                "start": metas[i]["start"],
                "snippet": metas[i]["snippet"],
            })
        sizes = [0] * k
        for lbl in labels:
            sizes[int(lbl)] += 1
        # Пустые кластеры (KMeans изредка их оставляет) → фантомные темы
        # размера 0; не показываем их в легенде.
        topics = [
            {"cluster": c, "label": label_map.get(c, f"тема {c + 1}"),
             "color": palette[c], "size": sizes[c]}
            for c in range(k) if sizes[c] > 0
        ]

        _cache = _MapCache(points, topics, count, int(vectors.shape[1]), kind)
        logger.info("Embedding map built: %d points, %d topics (%s)", n, len(topics), kind)
        return _serialize(_cache)


def locate(query: str, top_k: int = 5) -> dict:
    """Спроецировать запрос в текущую карту + вернуть id ближайших чанков.

    - highlight: точные соседи через near_text Weaviate (как в реальном RAG).
    - marker: центроид уже спроецированных координат этих соседей. Так маркер
      гарантированно стоит СРЕДИ своих же подсвеченных точек (никакого расхождения
      из-за отдельного эмбеддинга/UMAP.transform). None, если соседей нет в карте.
    """
    global _cache
    from system.llm.llm_services import get_chat_vectore_store_manager
    if _cache is None:
        build_map()
    with _lock:
        cache = _cache
    manager = get_chat_vectore_store_manager()

    # Точные соседи (для подсветки) — те же объекты, что вернул бы поиск.
    highlight_ids: List[str] = []
    try:
        resp = manager.collection.query.near_text(query=query, limit=top_k)
        highlight_ids = [str(o.uuid) for o in resp.objects]
    except Exception as e:  # noqa: BLE001
        logger.warning("near_text для locate не удался: %s", e)

    # Маркер = центр масс подсвеченных точек, присутствующих в карте.
    marker = None
    if cache is not None and cache.points and highlight_ids:
        idset = set(highlight_ids)
        coords = [(p["x"], p["y"], p["z"]) for p in cache.points if p["id"] in idset]
        if coords:
            m = np.asarray(coords, dtype=np.float32).mean(axis=0)
            marker = {"x": round(float(m[0]), 3),
                      "y": round(float(m[1]), 3),
                      "z": round(float(m[2]), 3)}

    return {"highlight_ids": highlight_ids, "marker": marker}


def reset_cache() -> None:
    """Сбросить кэш (для тестов / принудительной пересборки)."""
    global _cache
    with _lock:
        _cache = None
