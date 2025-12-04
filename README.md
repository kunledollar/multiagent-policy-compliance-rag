# Enterprise Policy & Compliance Assistant — Multi-Agent RAG System

End-to-end Retrieval-Augmented Generation (RAG) platform for policy and compliance assistance.

## Features

- Multi-agent orchestration (retrieval, summarisation, reasoning, fact-checking, answer writing)
- FastAPI backend `/health`, `/ingest`, `/query`
- Streamlit dashboard for business users
- FAISS vector store
- TXT / PDF / DOCX ingestion
- Docker + docker-compose deployment
- Nginx virtual hosts:
  - `rag_api.kunledollar.com` → API
  - `rag_dashboard.kunledollar.com` → Streamlit
  - `rag_grafana.kunledollar.com` → Grafana
- Prometheus metrics + Grafana dashboards
- GitHub Actions CI/CD to EC2

## Local Docker

```bash
cp .env.example .env
# edit .env to add OPENAI_API_KEY

docker compose up -d --build
```

Put your policy documents under `data/raw/` before ingestion.

## CI/CD

- Push this repo to GitHub.
- Add repo secrets:

  - `EC2_HOST` (e.g. `ubuntu@3.136.159.224`)
  - `EC2_SSH_KEY` (contents of rag_key.pem)
  - `EC2_PROJECT_DIR` (e.g. `/home/ubuntu/enterprise_policy_compliance_rag`)
  - `OPENAI_API_KEY`

On push to `main` the workflow will deploy to EC2 and run `docker compose up -d --build`.
CI/CD pipeline test at Thu Dec  4 13:12:47 UTC 2025
