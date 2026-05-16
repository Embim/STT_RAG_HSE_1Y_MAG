"""MLflow instrumentation for the ingest/benchmark/eval pipelines.

Three context managers, all fail-soft (if MLflow is disabled/unreachable they
yield a no-op object and downstream code keeps working):

  - `video_run(...)` — for ingest of one video. Starts a hardware sidecar
    (nvidia-ml-py + psutil) that writes GPU/CPU/RAM metrics every 5 seconds.
  - `tracked_run(...)` — generic single run for benchmark/eval/anything else
    that doesn't need the hardware sidecar.
  - `batch_run(...)` — parent run aggregating progress across many child runs.
    Returns a handle with `.update_progress(done, failed)` that maintains
    `progress_pct`, `eta_minutes`, `avg_seconds_per_job` on the parent run via
    direct MlflowClient calls (works across separate Python processes /
    Airflow tasks — children just need to read the parent run_id from xcom).

All runs share fail-soft semantics: missing libs, no tracking URI, server down
→ noop, no exceptions propagate.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# Lazy imports so this module is importable without the `ops` extra installed.
def _import_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        return mlflow, MlflowClient
    except ImportError:
        return None, None


def _import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def _import_pynvml():
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
        except Exception as e:  # no GPU, driver missing, etc.
            logger.debug("nvidia-ml-py init failed: %s", e)
            return None
        return pynvml
    except ImportError:
        return None


def _tracking_uri() -> Optional[str]:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        from settings import settings
        return settings.MLFLOW_TRACKING_URI or None
    except Exception:
        return None


def _experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME") or "stt-rag-ingest"


@contextmanager
def mlflow_span(
    name: str,
    *,
    span_type: str = "CHAIN",
    inputs: Optional[dict] = None,
    attributes: Optional[dict] = None,
) -> Iterator[Any]:
    """MLflow Tracing v2.14+ обёртка с fail-soft семантикой.

    Использование:
        with mlflow_span("judge_eval", span_type="AGENT", inputs={...}) as span:
            ...
            with mlflow_span(f"chunk_{i:03d}", span_type="LLM", inputs={...}) as cs:
                cs.set_outputs({"findings": [...]})  # после LLM call

    Spans логируются в активный run и видны во вкладке Traces в MLflow UI.
    Span_type — один из CHAIN/AGENT/LLM/CHAT_MODEL/RETRIEVER/TOOL/PARSER/UNKNOWN.

    Если mlflow не установлен или нет active run — yield-им stub-объект с
    `.set_outputs(...)` no-op. Downstream код работает без знания о MLflow.
    """
    mlflow, _ = _import_mlflow()

    class _StubSpan:
        def set_outputs(self, *args, **kwargs) -> None: ...
        def set_attributes(self, *args, **kwargs) -> None: ...
        def set_inputs(self, *args, **kwargs) -> None: ...

    if mlflow is None:
        yield _StubSpan()
        return
    try:
        # MLflow Tracing появился в 2.14. Если установлен старший — fallback.
        start_span = getattr(mlflow, "start_span", None)
        if start_span is None:
            yield _StubSpan()
            return
        ctx = start_span(name=name, span_type=span_type,
                         inputs=inputs, attributes=attributes)
    except Exception as e:
        logger.debug("mlflow_span(%s) start failed: %s", name, e)
        yield _StubSpan()
        return

    try:
        with ctx as span:
            yield span if span is not None else _StubSpan()
    except Exception as e:
        logger.debug("mlflow_span(%s) failed inside: %s", name, e)


class _NoopRun:
    """Stand-in when MLflow is disabled — same API, does nothing."""

    run_id: Optional[str] = None

    def log_progress(self, pct: float, step: Optional[int] = None) -> None: ...
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None: ...
    def log_param(self, name: str, value: Any) -> None: ...
    def log_artifact(self, local_path: str) -> None: ...
    def log_table(self, data: Any, artifact_file: str) -> None: ...
    def set_tag(self, name: str, value: str) -> None: ...


class _ActiveRun:
    """Live MLflow run — wraps `mlflow.log_*` so the caller never has to know."""

    def __init__(self, mlflow_module, run_id: str) -> None:
        self._mlflow = mlflow_module
        self.run_id = run_id

    def log_progress(self, pct: float, step: Optional[int] = None) -> None:
        try:
            self._mlflow.log_metric("progress_pct", float(pct), step=step)
        except Exception as e:
            logger.debug("log_progress failed: %s", e)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        try:
            self._mlflow.log_metric(name, float(value), step=step)
        except Exception as e:
            logger.debug("log_metric(%s) failed: %s", name, e)

    def log_param(self, name: str, value: Any) -> None:
        try:
            self._mlflow.log_param(name, value)
        except Exception as e:
            logger.debug("log_param(%s) failed: %s", name, e)

    def log_artifact(self, local_path: str) -> None:
        if not local_path or not Path(local_path).exists():
            return
        try:
            self._mlflow.log_artifact(local_path)
        except Exception as e:
            logger.debug("log_artifact(%s) failed: %s", local_path, e)

    def log_table(self, data: Any, artifact_file: str) -> None:
        """Залить pandas-таблицу как JSON-артефакт; UI отрендерит как sortable table.

        `data` — pandas.DataFrame или dict-of-lists. `artifact_file` — путь
        относительно run-а, обычно `findings.json` или `tables/foo.json`.
        Требует mlflow>=2.8 (у нас в eval extras >=2.18).
        """
        try:
            self._mlflow.log_table(data=data, artifact_file=artifact_file)
        except Exception as e:
            logger.debug("log_table(%s) failed: %s", artifact_file, e)

    def set_tag(self, name: str, value: str) -> None:
        try:
            self._mlflow.set_tag(name, value)
        except Exception as e:
            logger.debug("set_tag(%s) failed: %s", name, e)


class _HardwareSidecar(threading.Thread):
    """Background thread that samples GPU/CPU/RAM and pushes them to the run."""

    def __init__(self, client, run_id: str, *, interval: float = 5.0) -> None:
        super().__init__(daemon=True, name="mlflow-hw-sidecar")
        self._client = client
        self._run_id = run_id
        self._interval = interval
        self._stop = threading.Event()
        self._psutil = _import_psutil()
        self._pynvml = _import_pynvml()
        self._gpu_handles = []
        if self._pynvml is not None:
            try:
                count = self._pynvml.nvmlDeviceGetCount()
                self._gpu_handles = [
                    self._pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)
                ]
            except Exception as e:
                logger.debug("nvml device enumeration failed: %s", e)
                self._pynvml = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        step = 0
        while not self._stop.is_set():
            ts = int(time.time() * 1000)
            try:
                self._sample(ts, step)
            except Exception as e:
                logger.debug("hw sidecar sample failed: %s", e)
            step += 1
            self._stop.wait(self._interval)

    def _sample(self, ts: int, step: int) -> None:
        from mlflow.entities import Metric  # local import — only needed when active

        metrics: list = []

        if self._psutil is not None:
            cpu_pct = self._psutil.cpu_percent(interval=None)
            ram = self._psutil.virtual_memory()
            metrics.append(Metric("cpu_pct", float(cpu_pct), ts, step))
            metrics.append(Metric("ram_used_gb", ram.used / 1024**3, ts, step))
            metrics.append(Metric("ram_pct", float(ram.percent), ts, step))

        if self._pynvml is not None:
            for idx, h in enumerate(self._gpu_handles):
                util = self._pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(h)
                temp = self._pynvml.nvmlDeviceGetTemperature(h, 0)  # 0 = GPU
                suffix = "" if len(self._gpu_handles) == 1 else f"_gpu{idx}"
                metrics.append(Metric(f"gpu_util_pct{suffix}", float(util.gpu), ts, step))
                metrics.append(Metric(f"gpu_mem_used_gb{suffix}", mem.used / 1024**3, ts, step))
                metrics.append(Metric(f"gpu_temp_c{suffix}", float(temp), ts, step))

        if not metrics:
            return
        # Batch-write so the sidecar makes ~1 HTTP call per cycle, not 7.
        self._client.log_batch(self._run_id, metrics=metrics)


def _ensure_mlflow_session():
    """Common setup. Returns (mlflow_module, MlflowClient_class, uri) or (None, None, None)."""
    mlflow, MlflowClient = _import_mlflow()
    uri = _tracking_uri()
    if mlflow is None or not uri:
        logger.info("MLflow disabled (no module or no tracking URI); progress not tracked")
        return None, None, None
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(_experiment_name())
    except Exception as e:
        logger.warning("Cannot configure MLflow: %s — falling back to noop", e)
        return None, None, None
    return mlflow, MlflowClient, uri


@contextmanager
def video_run(
    *,
    title: str,
    source_file_name: Optional[str] = None,
    asr_name: Optional[str] = None,
    audio_size_mb: float = 0.0,
    chunk_minutes: int = 0,
    extra_params: Optional[dict] = None,
    sidecar_interval: float = 5.0,
    parent_run_id: Optional[str] = None,
) -> Iterator[Any]:
    """Ingest run with hardware sidecar.

    If `parent_run_id` is provided, the child run gets a tag pointing back to
    the batch parent so MLflow UI can navigate parent ↔ children.
    """
    mlflow, MlflowClient, uri = _ensure_mlflow_session()
    if mlflow is None:
        yield _NoopRun()
        return

    try:
        run = mlflow.start_run(run_name=title)
    except Exception as e:
        logger.warning("Cannot start MLflow run: %s — falling back to noop", e)
        yield _NoopRun()
        return

    run_id = run.info.run_id
    client = MlflowClient(tracking_uri=uri)
    sidecar = _HardwareSidecar(client, run_id, interval=sidecar_interval)
    active = _ActiveRun(mlflow, run_id)
    try:
        active.set_tag("kind", "ingest")
        if source_file_name:
            active.log_param("source_file_name", source_file_name)
            active.set_tag("source_file_name", source_file_name)
        if asr_name:
            active.log_param("asr_name", asr_name)
        if audio_size_mb:
            active.log_param("audio_size_mb", round(audio_size_mb, 2))
        if chunk_minutes:
            active.log_param("chunk_minutes", chunk_minutes)
        if parent_run_id:
            active.set_tag("parent_batch_run_id", parent_run_id)
        for k, v in (extra_params or {}).items():
            active.log_param(k, v)
        sidecar.start()
        yield active
    finally:
        sidecar.stop()
        sidecar.join(timeout=2.0)
        try:
            mlflow.end_run()
        except Exception as e:
            logger.debug("end_run failed: %s", e)


@contextmanager
def tracked_run(
    *,
    title: str,
    kind: str,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    parent_run_id: Optional[str] = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Generic single MLflow run (no hardware sidecar).

    Use for benchmark / eval / anything that doesn't need GPU sampling.

    Args:
        nested: если True, открыть как nested-run внутри уже активного parent run.
            Нужно для sweep-сценария: parent batch run открыт в этом же процессе,
            children — nested. Видны в UI как drilldown дерево.
            По умолчанию False — `mlflow.start_run` без nested будет падать
            если уже есть active run, поэтому ставь True только сознательно.
    """
    mlflow, _, _ = _ensure_mlflow_session()
    if mlflow is None:
        yield _NoopRun()
        return

    try:
        run = mlflow.start_run(run_name=title, nested=nested)
    except Exception as e:
        logger.warning("Cannot start MLflow run: %s — falling back to noop", e)
        yield _NoopRun()
        return

    active = _ActiveRun(mlflow, run.info.run_id)
    try:
        active.set_tag("kind", kind)
        if parent_run_id:
            active.set_tag("parent_batch_run_id", parent_run_id)
        for k, v in (params or {}).items():
            active.log_param(k, v)
        for k, v in (tags or {}).items():
            active.set_tag(k, str(v))
        yield active
    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            logger.debug("end_run failed: %s", e)


class _BatchHandle:
    """Manages a parent batch run — updates aggregate progress/ETA from outside.

    Designed to survive across processes: the run_id is the contract, anyone
    holding it can update parent metrics via MlflowClient without an active
    `start_run` context. Use this when child jobs run as separate Airflow tasks
    or subprocesses.
    """

    def __init__(self, mlflow_module, client, run_id: str, n_jobs: int) -> None:
        self._mlflow = mlflow_module
        self._client = client
        self.run_id = run_id
        self._n_jobs = max(1, n_jobs)
        self._started_at = time.time()
        # Set by open_batch_run; finalize нужен чтобы в nested-mode закрыть
        # parent на стороне локального процесса (active run).
        self._nested = False

    def update_progress(self, *, done: int, failed: int = 0, step: Optional[int] = None) -> None:
        try:
            ts = int(time.time() * 1000)
            elapsed = time.time() - self._started_at
            avg_sec = elapsed / max(1, done) if done > 0 else 0.0
            remaining = max(0, self._n_jobs - done) * avg_sec
            metrics = [
                self._mlflow.entities.Metric("n_done", float(done), ts, step or done),
                self._mlflow.entities.Metric("n_failed", float(failed), ts, step or done),
                self._mlflow.entities.Metric(
                    "progress_pct", 100.0 * done / self._n_jobs, ts, step or done
                ),
                self._mlflow.entities.Metric("avg_sec_per_job", avg_sec, ts, step or done),
                self._mlflow.entities.Metric("eta_minutes", remaining / 60, ts, step or done),
            ]
            self._client.log_batch(self.run_id, metrics=metrics)
        except Exception as e:
            logger.debug("batch update_progress failed: %s", e)

    def finalize(self, *, n_succeeded: int, n_failed: int) -> None:
        try:
            self._client.set_tag(self.run_id, "n_succeeded", str(n_succeeded))
            self._client.set_tag(self.run_id, "n_failed_final", str(n_failed))
            self._client.set_tag(
                self.run_id, "wall_minutes", f"{(time.time() - self._started_at) / 60:.1f}"
            )
        except Exception as e:
            logger.debug("batch finalize failed: %s", e)


def open_batch_run(
    *,
    name: str,
    kind: str = "batch",
    n_jobs: int = 0,
    params: Optional[dict] = None,
    nested: bool = False,
) -> Optional[_BatchHandle]:
    """Open a parent batch run that will be updated from outside or via children.

    Args:
        nested: если True, parent остаётся активным как top-level run в этом
            процессе. Любые `mlflow.start_run(nested=True)` будут вложены под
            него (видно в UI как раскрывающееся дерево). Используется для
            sweep-скриптов, где parent и children живут в одном процессе.
            По умолчанию False — старый detach mode: parent отвязывается через
            `end_run(status="RUNNING")` чтобы subsequent `start_run` в этом
            процессе не вложился случайно. Подходит для cross-process сценариев
            (Airflow tasks, subprocess), где детям нужен только run_id.

    Returns the handle (with `run_id`) or None if MLflow is disabled. Caller
    is responsible for calling `.finalize_batch_run(...)` later.
    """
    mlflow, MlflowClient, uri = _ensure_mlflow_session()
    if mlflow is None:
        return None

    try:
        run = mlflow.start_run(run_name=name)
    except Exception as e:
        logger.warning("Cannot start batch run: %s", e)
        return None

    run_id = run.info.run_id
    client = MlflowClient(tracking_uri=uri)
    try:
        client.set_tag(run_id, "kind", kind)
        client.log_param(run_id, "n_jobs", n_jobs)
        for k, v in (params or {}).items():
            client.log_param(run_id, k, str(v))
    except Exception as e:
        logger.debug("batch tagging failed: %s", e)

    if not nested:
        # Detach: end active context so children в other processes/tasks
        # не наследуют parent как nested случайно. Сам batch run остаётся
        # OPEN на стороне MLflow до finalize_batch_run.
        try:
            mlflow.end_run(status="RUNNING")
        except Exception:
            pass
    # nested=True: parent остаётся active в этом процессе. finalize_batch_run
    # потом сам сделает end_run.

    handle = _BatchHandle(mlflow, client, run_id, n_jobs)
    handle._nested = nested  # для finalize: знать, надо ли локально end_run
    return handle


def reattach_batch(run_id: str, n_jobs: int) -> Optional[_BatchHandle]:
    """Get a handle for an already-open batch run by id (used in child tasks)."""
    mlflow, MlflowClient, uri = _ensure_mlflow_session()
    if mlflow is None or not run_id:
        return None
    try:
        client = MlflowClient(tracking_uri=uri)
        return _BatchHandle(mlflow, client, run_id, n_jobs)
    except Exception as e:
        logger.debug("reattach_batch failed: %s", e)
        return None


def finalize_batch_run(handle: Optional[_BatchHandle], *, n_succeeded: int, n_failed: int) -> None:
    """Close out a batch run by writing final tags and ending it in MLflow."""
    if handle is None:
        return
    handle.finalize(n_succeeded=n_succeeded, n_failed=n_failed)
    # В nested-mode parent ещё активен в этом процессе — закроем локально
    # через mlflow.end_run. Это и поставит run в FINISHED на сервере.
    if getattr(handle, "_nested", False):
        try:
            handle._mlflow.end_run(status="FINISHED")
            return
        except Exception as e:
            logger.debug("nested end_run failed: %s", e)
    try:
        handle._client.set_terminated(handle.run_id, status="FINISHED")
    except Exception as e:
        logger.debug("set_terminated failed: %s", e)
