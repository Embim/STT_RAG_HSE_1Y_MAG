# Custom Airflow image for the STT_RAG project.
#
# - Base: official apache/airflow 2.10 with Python 3.12 (compatible with our
#   pyproject.toml requires-python>=3.12 and the existing project deps).
# - Adds: project deps that DAGs need to import (mlflow, boto3, nvidia-ml-py,
#   weaviate, langchain-core, langfuse, pydantic, openai, ...).
# - Adds: static ffmpeg+ffprobe binaries from mwader/static-ffmpeg via
#   multi-stage copy. Skips `apt-get install ffmpeg` because the deb.debian.org
#   CDN (Fastly) is flaky from many Russian ISPs — apt-get hangs at random
#   package downloads. Static binaries are ~75 MB vs 432 MB of apt deps.
# - Adds: PYTHONPATH=/opt/airflow/project/src so DAGs can `import evaluation.*`,
#   `import system.*`, etc. via the volume-mounted code.
FROM mwader/static-ffmpeg:7.0 AS ffmpeg-bin

FROM apache/airflow:2.10.3-python3.12

USER root
COPY --from=ffmpeg-bin /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg-bin /ffprobe /usr/local/bin/ffprobe
# Verify both binaries run; redirect stdout to /dev/null instead of piping
# to `head -1` — under bash `pipefail` the early pipe close sends SIGPIPE
# to ffmpeg on its second write and the whole RUN exits 141.
RUN chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && /usr/local/bin/ffmpeg  -version > /dev/null \
    && /usr/local/bin/ffprobe -version > /dev/null

USER airflow
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
# files.pythonhosted.org через Fastly бывает медленный из РФ — без длинного
# таймаута и ретраев build падает на крупных пакетах (torch-cpu, mlflow).
RUN pip install --no-cache-dir \
        --timeout 180 \
        --retries 8 \
        -r /tmp/requirements-airflow.txt

ENV PYTHONPATH=/opt/airflow/project/src
ENV TZ=UTC
