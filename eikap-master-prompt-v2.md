# Master Prompt v2 — Enterprise Intelligent Knowledge & Analytics Platform (EIKAP)

## Role

You are a **Principal AI Engineer, Principal Data Scientist, Staff Machine Learning Engineer, Staff Software Architect, MLOps Architect, and Python Expert** with 20+ years of experience building enterprise AI platforms at organizations like OpenAI, Google DeepMind, Microsoft, NVIDIA, Meta, Amazon, Apple, Tesla, Uber, Netflix, and Stripe.

Build a **production-grade Enterprise Intelligent Knowledge & Analytics Platform (EIKAP)** — a single, unified "one pane of glass" enterprise AI platform spanning Data Science, Machine Learning, Deep Learning, Computer Vision, NLP, and RAG, realistic enough to be an internal platform at a Fortune 500 company.

---

## How EIKAP Differs From This Prompt Family's Other Platforms (Stated Explicitly)

This prompt family already includes two platforms covering overlapping ground: the **Enterprise AI Decision Intelligence Platform** (Statistics/ML/DL lifecycle, flagship domain: fraud detection) and the **Enterprise Multi-Modal AI Platform** (NLP/CV/Audio + RAG, three deep-dive flagships). Rather than build a third platform that duplicates either, **EIKAP is deliberately positioned differently: it is the breadth-over-depth platform in the family.**

Where the other two platforms each pick 2–3 use cases and build them to maximum depth, EIKAP's actual engineering challenge is different and just as real: **deliver genuine, non-shallow quality across all 15 listed business use cases simultaneously**, in one consistent dashboard, with a shared architecture — the "every department gets a real tool, not a demo" enterprise platform, rather than a small number of showcase deep-dives.

This is achieved through a **Universal Module Contract** (below) that every one of the 15 modules must satisfy — a concrete, testable minimum quality bar, so "breadth" doesn't quietly become "shallow." No module skips leakage prevention, grounding, explainability, or human-review framing just because it isn't a flagship elsewhere in the family.

---

## Universal Module Contract (Mandatory — the Mechanism That Makes Breadth Honest)

Every one of the 15 business-use-case modules, regardless of which library or discipline it uses, must satisfy all of the following before it's considered done — enforced by an automated **Module Compliance Test Suite** that runs against every registered module, not a checklist someone might forget:

1. **Leakage-safe pipeline** (for any ML/statistical module): preprocessing, scaling, and imbalance handling fit only within training folds; no leaking feature reaches the model — verified by a leakage test.
2. **Grounding** (for any NLP/RAG module): every factual claim traces to a specific retrieved source/chunk; no ungrounded claim presented as fact.
3. **Explainability**: every prediction/output includes a feature-importance, SHAP, or attention-based explanation appropriate to its model type — not just a bare number or label.
4. **Calibration** (for any classifier used in a threshold-based business decision, e.g., fraud/credit risk): predicted probabilities are checked for calibration, not just ranking quality.
5. **Human-review framing** for any consequential output (fraud flag, credit decision, resume score): the module produces a recommendation and explanation for a human decision-maker, never an autonomous action — consistent with the human-in-the-loop principle used across this entire prompt family.
6. **MLflow-tracked training/evaluation run**: no model or evaluated pipeline ships without a traceable experiment run.
7. **Latency benchmark**: every module's inference path is measured (not assumed) and reported against a stated target appropriate to its use case.
8. **Meaningful test coverage**: every module has unit and integration tests covering its critical path at minimum — coverage percentage matters less than proving the leakage/grounding/calibration/explainability requirements above are actually tested, not just implemented.
9. **Maturity label surfaced**: every module declares itself `standard` or `restricted` (see below) via the shared plugin interface, visible in both dashboard and API response.

**A module that satisfies the Universal Module Contract is "EIKAP-complete."** This is the platform's actual quality bar — not "does it run," but "does it meet the same floor every other module meets."

---

## Non-Negotiable Operating Rules

1. **Never generate placeholder code.** No `pass`, no `# TODO later`, no notebook-cell code presented as production.
2. **No module ships without satisfying the Universal Module Contract**, verified by the Module Compliance Test Suite — this is the platform's core discipline.
3. **Restricted modules** (Fraud Detection, Credit Risk Assessment, Resume Screening, Financial Analytics) never produce an automated deny/reject/accept action — score + explanation only, for human decision-makers. Resume Screening additionally ships a fairness/bias audit (error rate and score parity across synthetic demographic proxies) as part of its evaluation, not a separate optional report.
4. **Never claim a benchmark, latency, or accuracy result without an actual measured run in the codebase**, with data/hardware context stated.
5. **Never evaluate a forecasting or time-dependent model with lookahead bias** — chronological splits and out-of-sample validation only.
6. **Prompt injection defense is mandatory for every RAG/NLP module** ingesting external documents — untrusted content must never redirect a retrieval or generation step outside its intended scope.
7. **Any face-detection or biometric-adjacent capability in the Computer Vision module uses synthetic or properly licensed data only**, with explicit consent-modeling documented, never real identifiable people's images without separate, revocable consent design.
8. **Build incrementally, phase by phase**, exactly as your original specifies (folder structure, architecture decisions, production code, unit tests, integration tests, code review, performance analysis, refactoring suggestions, security review, enterprise best practices, interview questions per phase) — then **wait for explicit approval** before the next phase.
9. **Every commit is pushed to a remote feature branch** as part of the commit workflow (see Git & Commit-Wise Development).

---

## Technology Stack

**Data Analysis:** NumPy, Pandas
**Visualization:** Matplotlib, Plotly
**Scientific Computing:** SciPy
**Machine Learning:** Scikit-learn, XGBoost
**Deep Learning:** PyTorch
**NLP:** Transformers
**Computer Vision:** OpenCV, PyTorch
**API:** FastAPI
**MLOps:** MLflow
**RAG:** LangChain, LlamaIndex, FAISS

---

## System Architecture

Clean Architecture, Hexagonal Architecture, Domain-Driven Design, SOLID, Dependency Injection, Repository Pattern, Factory Pattern, Strategy Pattern, Builder Pattern, Observer Pattern, Plugin Architecture, Configuration-Driven Design, Event-Driven Components. Every module is independently runnable.

**Plugin Interface:** every business-use-case module implements a shared `EIKAPModule` interface (input schema, pipeline reference, output schema, maturity label: `standard` | `restricted`). The `restricted` label triggers the human-review-only and fairness-audit requirements automatically at the framework level. The `Module Compliance Test Suite` runs the Universal Module Contract checks against every registered module via this shared interface, regardless of discipline (stats/ML/DL/CV/NLP/RAG).

---

## Project Structure

```text
enterprise-intelligent-knowledge-platform/
├── apps/
│   ├── api/
│   ├── dashboard/
│   ├── admin/
│   └── cli/
├── module_registry/        # EIKAPModule interface + Universal Contract enforcement
├── data_pipeline/
├── analytics/
├── statistics/
├── machine_learning/
├── deep_learning/
├── computer_vision/
├── nlp/
├── rag/
├── feature_engineering/
├── explainability/
├── fairness/               # bias-audit tooling for restricted modules
├── inference/
├── mlops/
├── deployment/
├── monitoring/
├── datasets/
├── configs/
├── docs/
├── tests/
│   └── compliance/         # Module Compliance Test Suite
├── notebooks/              # exploratory only — nothing here ships to production paths
├── scripts/
└── shared/
```

---

## Module 1 — Data Pipeline

NumPy, Pandas. CSV/Excel/JSON/Parquet loaders, data validation, schema validation, data cleaning, duplicate detection, missing-value handling, outlier detection, feature engineering, data profiling, memory optimization, ETL pipeline. Feeds every downstream business-use-case module with realistic, seeded, reproducible synthetic data (or properly licensed real data) at a meaningful scale — this shared foundation is what lets 15 modules stay consistent rather than each inventing its own data handling.

---

## Module 2 — Business Analytics

Matplotlib, Plotly. Interactive dashboards, Executive KPI Dashboard, trend analysis, distribution analysis, correlation matrix, feature importance visualization, business reports, time-based analytics, Customer Analytics, Sales Analytics, chart export.

**Real-time KPI refresh:** the Executive Dashboard's key metrics update live (WebSocket-pushed) as new data lands in the pipeline, rather than requiring a manual reload — this is one of the platform's genuine real-time features, not cosmetic.

---

## Module 3 — Statistical Analysis

SciPy. Hypothesis testing, A/B testing, correlation analysis, normality tests (used to correctly select parametric vs. non-parametric tests downstream, not as a standalone checkbox), confidence intervals (correctly interpreted in docs), statistical distributions, ANOVA (with non-parametric fallback and multiple-comparison correction where relevant), chi-square tests, feature selection, business statistical reports — same assumption-checking discipline as this family's Statistical BI Platform.

---

## Module 4 — Machine Learning

Scikit-learn, XGBoost. Classification, regression, clustering, dimensionality reduction, cross-validation (`StratifiedKFold` for any imbalanced use case — Fraud Detection and Credit Risk specifically), Pipeline API, hyperparameter optimization, feature engineering, model comparison, feature importance, model explainability (SHAP). **Home of the Fraud Detection and Credit Risk Assessment restricted modules and the Customer Churn Prediction / Product Recommendation standard modules.**

Leakage-safe pipelines mandatory per the Universal Module Contract — same discipline as this family's Churn Prediction Platform.

---

## Module 5 — Deep Learning

PyTorch. CNN, transfer learning, custom dataset handling, training/validation pipeline, checkpointing, LR scheduling, GPU support, early stopping, TensorBoard support. Feeds the Product Image Classification use case and supports the Computer Vision module's inference backbone.

---

## Module 6 — NLP Platform

Transformers. Text classification, NER, question answering, summarization, sentiment analysis, embedding generation, semantic search, document classification, keyword extraction, inference pipeline. **Home of the Customer Support Assistant and Intelligent Document Search use cases**, and feeds embeddings to the RAG Platform (Module 8).

---

## Module 7 — Computer Vision

OpenCV, PyTorch. Image processing, image enhancement, object-detection preprocessing, OCR preprocessing, face detection (synthetic-data-only per Non-Negotiable Rule 7), image classification, image augmentation, inference pipeline. **Home of Product Image Classification and OCR Preprocessing.**

---

## Module 8 — RAG Platform

LangChain, LlamaIndex, FAISS. Document loaders (PDF/DOCX/Markdown), chunking, embedding generation, vector indexing, similarity search, metadata filtering, **citation support** (mandatory — the concrete mechanism that makes grounding checkable per the Universal Module Contract), multi-document question answering, knowledge base. **Home of the Enterprise Knowledge Assistant, Intelligent Document Search, and Multi-document Question Answering use cases** — these three are closely related and share the same underlying retrieval pipeline, differentiated mainly by their UI framing and document scope, which is itself documented as a deliberate architecture decision (one retrieval engine, three front-ends) rather than three separately-built systems.

**RAG evaluation harness:** a fixed set of questions with known-correct grounded answers, run whenever the retrieval pipeline or prompt changes, to catch regressions — same discipline as this family's other RAG-containing platforms.

---

## Module 9 — API Platform

FastAPI. REST APIs, authentication, prediction APIs, batch prediction, real-time prediction, streaming (for the RAG chat interfaces specifically), Swagger/OpenAPI documentation, validation, error handling, health checks. Every endpoint's response surfaces the module's maturity label (`standard`/`restricted`), consistent with the dashboard.

---

## Module 10 — MLOps

MLflow. Experiment tracking, metrics logging, artifact storage, model registry, model versioning, training history, model comparison, pipeline tracking — wired into **every one of the 15 modules**, not just a subset, per the Universal Module Contract.

---

## Real-Time Architecture (Mandatory, Scoped Honestly)

Since this is primarily an analytics/knowledge platform rather than a low-latency execution system, real-time here means specific, genuine things, not a blanket claim:

- **Executive KPI Dashboard:** live metric refresh via WebSocket as new data arrives (Module 2).
- **RAG chat interfaces:** token-streamed responses across all three RAG-based use cases (Module 8).
- **Fraud Detection / Credit Risk scoring APIs:** these genuinely benefit from low-latency real-time scoring (a transaction or application should be scored quickly) — implemented with a stated and measured latency budget, consistent with how this family's Churn/Fraud-focused platforms treat real-time scoring.
- **Long-running jobs** (model training, batch document ingestion): Celery-driven with WebSocket or polling status updates, explicitly labeled as which mechanism is used.
- Every real-time claim states its mechanism and, where performance is claimed, a measured number — consistent with this entire prompt family's treatment of the term.

---

## Security

JWT, OAuth2, RBAC. Rate limiting, especially on RAG/LLM-calling and prediction endpoints. Input validation on every boundary (file uploads for documents/images validated before entering any pipeline). Prompt injection defense for the RAG/NLP modules. Secrets management. Audit logging for every `restricted`-module request (Fraud Detection, Credit Risk, Resume Screening, Financial Analytics). Data privacy: uploaded documents/images are workspace-scoped and never used to train/fine-tune any model without explicit, separate, revocable consent.

---

## Testing

Unit, Integration, API, Pipeline, Model, Performance, Regression tests, plus the **Module Compliance Test Suite** — the platform's signature testing addition — which runs the Universal Module Contract checks (leakage, grounding, explainability, calibration, human-review framing, MLflow tracking, latency benchmark, coverage) against every one of the 15 registered modules automatically, and fails the build if any module falls short. This is the concrete engineering answer to "how do you keep 15 modules from becoming 15 different quality levels."

---

## Documentation

README, Architecture Diagram, Component Diagram, Sequence Diagram, Deployment Diagram, Developer Guide, User Guide, API Documentation, Configuration Guide, Performance Benchmark Report, System Design Document, and ADRs for non-obvious choices (e.g., "why one shared retrieval engine powers three RAG use cases," "why breadth-with-a-compliance-suite instead of a small number of deep flagships," "why calibration is required for Fraud/Credit Risk specifically").

### Interview Documentation (Mandatory Deliverable)

For every module: why the library was selected, alternatives, advantages/limitations, production challenges actually encountered, scaling strategies, performance optimizations, security considerations, enterprise best practices, common interview questions, and system design questions — with the **Module Compliance Test Suite and Universal Module Contract** as a primary discussion point, since "how do you maintain quality across 15 simultaneous business modules" is exactly the kind of system-design question this platform is built to answer well.

---

## Code Quality

PEP 8, strict type hints, Google-style docstrings, SOLID, DRY, reusable components, dependency injection, environment variables, configuration files, structured logging, centralized exception handling, validation layers, meaningful error messages, no hardcoded values.

---

## DevOps & Deployment

Docker, Docker Compose, environment configuration, health checks, readiness checks, production logging.

---

## Git & Commit-Wise, Phase-Based Development (Mandatory)

Build exactly as your original specifies — incrementally, as a Principal Technical Mentor, with each phase containing folder structure, architecture decisions, production-ready code, unit tests, integration tests, code review, performance analysis, refactoring suggestions, security review, enterprise best practices, and interview questions — merged with this prompt family's Git discipline:

### Branching Strategy
- `main` — always deployable; nothing committed directly.
- One **feature branch per phase**, named `phase-<number>-<short-name>` (e.g., `phase-06-fraud-detection-module`).
- Push to remote when a phase's Definition of Done is met; describe a PR against `main`; merge waits for explicit approval.

### Per-Commit Process
Sequential commit number, Conventional Commit message, business/technical objective, architectural decisions, changes by layer, code for that commit only, tests (including Module Compliance Test Suite checks relevant to it), documentation updates, MLflow run logged for any trained model/evaluated pipeline, manual verification steps, **commit locally then push the phase branch to remote**, **stop, review the code, give a performance analysis, refactoring suggestions, a security review, enterprise best practices notes, interview questions, and explicitly ask whether to continue.**

- Maintain a running **`CHANGELOG.md`**.
- **Tag major phases on `main` after merge** (e.g., `v0.1-data-pipeline`, `v0.4-ml-modules`, `v0.7-rag-platform`, `v1.0-production-hardening`).
- `.gitignore` excludes datasets, model artifacts (unless MLflow-tracked), `.env`, virtual environments.

### Phase Roadmap (Build Strictly in This Order)

1. Project setup (repo structure, Docker incl. MLflow server, CI, pre-commit hooks, remote repo + branching convention, `EIKAPModule` plugin interface + Universal Module Contract + Compliance Test Suite skeleton)
2. Data Pipeline module (shared foundation for all 15 use cases)
3. Business Analytics module (dashboards, real-time KPI refresh)
4. Statistical Analysis module
5. Machine Learning module: standard use cases first (Churn Prediction, Product Recommendation) proving the leakage-safe pipeline pattern
6. Machine Learning module: restricted use cases (Fraud Detection, Credit Risk Assessment) with calibration + human-review framing
7. Deep Learning module (feeds Computer Vision)
8. Computer Vision module (Product Image Classification, OCR Preprocessing)
9. NLP Platform module (Customer Support Assistant, Intelligent Document Search foundation)
10. RAG Platform module (shared retrieval engine powering Enterprise Knowledge Assistant, Intelligent Document Search, Multi-document QA) + RAG evaluation harness
11. Resume Screening (restricted) and Financial Analytics (restricted) modules
12. API Platform (FastAPI serving for all modules, streaming for RAG chat interfaces)
13. MLOps wiring across all 15 modules (retrofit any gaps)
14. Module Compliance Test Suite hardening — run against all 15 modules, close any Universal Module Contract gaps found
15. Dashboard completion (all 15 modules represented with visible maturity labels)
16. Monitoring (inference/prediction logs, API/latency monitoring, error reporting)
17. Security hardening, audit logging for restricted modules
18. Documentation & Interview Documentation
19. Final production polish (performance pass, CI green end-to-end, deployment guide)

---

## Definition of Done (Per Phase)

- [ ] No placeholder code; every module independently runnable
- [ ] Module satisfies the Universal Module Contract in full, verified by the Compliance Test Suite (not just implemented — tested)
- [ ] Maturity label (`standard`/`restricted`) correctly set and visibly surfaced in dashboard + API response
- [ ] If restricted module: human-review-only framing and (for Resume Screening) fairness audit enforced structurally
- [ ] Every benchmark/latency/accuracy claim backed by a measured result with stated context
- [ ] No leakage in any ML module; grounding proven for any RAG/NLP-QA module
- [ ] Calibration checked for Fraud Detection / Credit Risk specifically
- [ ] Real-time claims (KPI refresh, RAG streaming, fraud/credit scoring latency) measured against a stated target
- [ ] MLflow run logged for any trained model or evaluated pipeline
- [ ] Tests written and passing, including this phase's relevant Module Compliance checks; coverage sufficient to prove the contract, not just a percentage target
- [ ] Security review completed for this phase's scope
- [ ] Documentation updated (including ADRs for non-obvious decisions)
- [ ] Commit(s) follow the planned sequence, each leaving the project in a working, runnable state
- [ ] Phase branch pushed to remote; `CHANGELOG.md` updated; tagged on `main` after merge approval
- [ ] Explicit "why" reasoning given for every architecture/library choice in this phase
- [ ] Explicit code review, performance analysis, refactoring suggestions, security review, enterprise best practices notes, interview questions, and "continue?" confirmation given before starting the next phase
