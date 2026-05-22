# MLflow tracking server with Postgres backend + S3 artifact store.
#
# - Base: official ghcr.io/mlflow/mlflow:v2.18.0 image.
# - Adds psycopg2-binary for the Postgres backend store.
# - Adds boto3 for the S3-compatible artifact store (MinIO).
FROM ghcr.io/mlflow/mlflow:v2.18.0

RUN pip install --no-cache-dir psycopg2-binary>=2.9.9 boto3>=1.35.0
