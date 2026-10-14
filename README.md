# BERT-family sentiment analysis coursework

Collaborative coursework source for CSV extraction, text cleaning/tokenization, train/validation splitting, DistilBERT model factories, guarded inference, binary metrics, and optional database readiness. A complete training function is not implemented. The base distilbert-base-uncased checkpoint produces a newly initialized classification head; its output class indices are not validated sentiment predictions without a suitable fine-tuned checkpoint.

## Attribution and provenance

This is a curated derivative of [the original collaborative repository](https://github.com/prathmesh210/sentiment-analysis-bert-prathmesh-siddhart-maharshi), audited local branch feature-phase3 at commit3f348a4953a814806cadc7dd2da09404791895be. The source README credits Prathmesh, Siddhart (also spelled Siddharth in the original report), and Maharshi. Their contributions are preserved as collaborative attribution, not reassigned exclusively to one author. The user explicitly attested upload rights for the candidate. No permissive license is assigned; source rights remain reserved. Original Git history and private project-board screenshot/reports are excluded.

## Setup

Python3.12 is the documented container/test target. Use an isolated environment and review requirements.txt before deciding to install dependencies. No packages were installed or locked during this preparation. Torch/Transformers, data libraries, safetensors and optional psycopg2-binary are declared; compatibility resolution, Docker build and full scientific tests were not run.

```powershell
python -B -m unittest discover -s tests_stdlib -v
python -m src.inference --help
```

Importing inference/model/database helper modules does not download a model, probe CUDA or connect to a database. Model factories are local-cache-only by default, disable remote Python model code, and require safetensors for model weights. Choosing a checkpoint does not prove its training data, quality or model rights; review its exact model card/license independently.

Inference is explicit:

```powershell
python -m src.inference --allow-inference --checkpoint <reviewed-local-checkpoint> --text "synthetic example"
```

No download occurs unless the separate --allow-download switch is chosen. Text is limited to10000characters and tokenization512tokens; default device is CPU. Real inference/model downloads were not run. An untrained base checkpoint is an educational architecture example, not a meaningful sentiment service.

The optional database helper requires --allow-connection and explicit DB_USER/DB_PASSWORD/DB_NAME environment values; .env.example contains only blanks/placeholders and is not loaded automatically. It masks connection error details and uses bounded retry/connection timeouts. Docker Compose publishes only loopback database access and requires explicit values; the database has its own optional profile. The app container's default is CLI help, not inference. No container or database was started.

## Tests and limitations

Seven standard-library tests cover metrics, invalid lengths/labels, inert defaults, mocked model factory/inference behavior, DB refusal/error redaction and CLI refusal. They require no Torch, models, database or network. Original pytest data-processing/CSV tests are retained but were not run because scientific dependencies were not installed. Cached-model tests require explicit BERT_TEST_CACHED_MODELS=1; pytest config forces Hugging Face offline mode. Duplicate test function definitions were removed while preserving an explicit cached forward-shape check. No real model accuracy is claimed.

Original dataset and generated artifacts are excluded. Create your own synthetic CSV with text,label columns; train/validation label conversion expects numeric0/1. The original cleaner intentionally strips non-ASCII letters and is not multilingual-safe. The original sample-loading helper and misleading training instructions were replaced by this documentation. Automatic DockerHub publishing workflows were excluded; manual source checks do not deploy or push containers.
