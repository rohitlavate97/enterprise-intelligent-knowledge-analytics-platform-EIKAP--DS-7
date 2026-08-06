# Changelog

All notable changes to EIKAP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-08-07

### Added
- Recommendation Engine module with `CollaborativeRecommender` (TruncatedSVD matrix factorization, User-based/Item-based CF)
- `ContentBasedRecommender` (TF-IDF metadata similarity embeddings & user profile feature aggregation)
- `HybridRecommender` (Standard maturity — weighted CF + CB score fusion, cold-start popularity fallback mitigation)
- `RecommendationEvaluator` (Precision@K, Recall@K, MAP@K, NDCG@K ranking evaluation metrics)
- `RecommendationModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for recommendation module (148 total platform tests passing)

## [0.8.0] - 2026-08-07

### Added
- Computer Vision & Image Analytics module with `ImagePreprocessor` (resizing, grayscale, ImageNet normalization, Gaussian blur, Otsu thresholding)
- `VisionFeatureExtractor` using pre-trained Torchvision backbones (ResNet18 / MobileNetV3), color histograms, and Canny edge detection
- `ProductDefectInspector` (Standard maturity — manufacturing scratch, structural anomaly, discoloration detection with bounding box localization)
- `DocumentOCRParser` (Standard maturity — binarization, contour text region detection, receipt parsing extracting merchant, date, total, line items)
- `ComputerVisionModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for computer vision module (138 total platform tests passing)

## [0.7.0] - 2026-08-07

### Added
- Natural Language Processing & Text Analytics module with `TextPreprocessor` and `TextEmbedder` (TF-IDF & dense embeddings)
- `TopicModeler` implementing Scikit-Learn LDA & NMF topic extraction
- `SentimentAnalyzer` (Standard maturity — VADER/lexicon hybrid sentiment and aspect-based sentiment)
- `ResumeScreener` (Restricted maturity — skill matching, job scoring, fairness audit check for gender/age/location proxies, recruiter-framed recommendations)
- `SupportTicketClassifier` (Standard maturity — category routing, urgency scoring, error/account entity extraction)
- `NLPModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for NLP module (128 total platform tests passing)

## [0.6.0] - 2026-08-07

### Added
- Deep Learning module with `TabularDataset` and `create_data_loaders` for PyTorch tabular pipelines
- PyTorch Neural Network architectures: `TabularMLP`, `TabularResNet`, and `CategoricalEmbeddingNet` (Entity Embeddings)
- `PyTorchTrainer` with `EarlyStopping`, LR scheduling (`ReduceLROnPlateau`), CPU/CUDA auto-device management, and model checkpointing
- `DeepLearningModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for deep learning (115 total platform tests passing)

## [0.5.0] - 2026-08-06

### Added
- Machine Learning module with `BaseMLModel` ABC, joblib model persistence, and MLflow experiment tracking
- `CustomerChurnModel` (Standard maturity — XGBoost/RandomForest churn prediction with top risk factor extraction)
- `FraudDetectionModel` (Restricted maturity — LightGBM/XGBoost fraud detection framed strictly as human-investigator recommendations)
- `CreditRiskModel` (Restricted maturity — Probability-calibrated classifier using `CalibratedClassifierCV` with ECE verification)
- `MLEvaluator` computing classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score, ECE) and regression metrics (RMSE, MAE, R2)
- `HyperparameterTuner` supporting GridSearch & RandomSearch cross-validation
- `MachineLearningModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for machine learning (104 total platform tests passing)

## [0.4.0] - 2026-08-06

### Added
- Statistical Analysis module with `HypothesisTester` supporting t-tests, ANOVA, Chi-square, Mann-Whitney U, Kruskal-Wallis, Wilcoxon
- `NormalityTester` with Shapiro-Wilk, D'Agostino-Pearson, KS tests, and parametric vs non-parametric test auto-selection
- `CorrelationAnalyzer` for Pearson, Spearman, Kendall tau, and Partial correlation
- `DistributionFitter` for distribution fitting (norm, expon, lognorm, gamma), AIC/BIC goodness-of-fit, and confidence intervals
- `StatisticalFeatureSelector` for ANOVA F-classif, Chi2, Mutual Info, and VIF multicollinearity scoring
- `StatisticalReportGenerator` generating Markdown statistical summary reports
- `StatisticalAnalysisModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for statistical analysis (89 total platform tests passing)

## [0.3.0] - 2026-08-06

### Added
- Business Analytics module with Matplotlib & Plotly dual-engine chart rendering
- Chart components: `TrendChart`, `DistributionChart`, `CorrelationChart`, `FeatureImportanceChart`, and `ChartExporter` (PNG, SVG, HTML, base64)
- `KPIDashboard` computing SaaS/Enterprise metrics: MRR, ARR, CAC, LTV, LTV:CAC, ARPU, Churn Rate, Conversion Rate, NPS
- `CustomerAnalytics` with RFM segmentation, cohort retention matrix, and CLV distribution
- `SalesAnalytics` with revenue breakdown by region/category, sales rep performance, discount impact, and funnel analysis
- `ExecutiveReportGenerator` producing GitHub-flavored Markdown and standalone styled HTML reports
- Real-time KPI refresh engine: `KPIStreamer` WebSocket publisher and `LiveKPIPublisher`
- `BusinessAnalyticsModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for analytics (69 total platform tests passing)

## [0.2.0] - 2026-08-06

### Added
- Data Pipeline module with `BaseLoader` ABC and `LoaderFactory` (CSV/TSV, Excel, JSON/JSONL, Parquet)
- Data validation framework: `DataValidator` and `SchemaValidator` with schema inference and serialization
- Data cleaning suite: `DuplicateDetector` (exact & fuzzy matching), `MissingValueHandler`, `OutlierDetector` (IQR, Z-score, modified Z-score)
- Data profiling module: `DataProfiler` for numeric, categorical, temporal statistics and correlation matrix
- Feature engineering: `FeatureTransformer` for date, interaction, polynomial, binned, text, and ratio features
- Memory optimization: `MemoryOptimizer` for numeric downcasting and string-to-category conversion
- ETL pipeline orchestrator: `ETLPipeline`, `PipelineStep`, `PipelineResult`, `PipelineBuilder`
- 15 reproducible, realistic synthetic data generators matching all platform use cases
- `DataPipelineModule` integration with `EIKAPModule` interface
- Unit & compliance test suite for data pipeline (51 tests total passing)

## [0.1.0] - 2026-08-06

### Added
- Initial project structure and repository setup
- EIKAPModule plugin interface with maturity labels (standard/restricted)
- Universal Module Contract with 10 automated compliance checks
- Module Compliance Test Suite skeleton
- Shared infrastructure: configuration, logging, exceptions, DI container
- Docker Compose setup with PostgreSQL, Redis, MLflow server
- GitHub Actions CI pipeline with lint, test, and compliance jobs
- Pre-commit hooks (ruff, mypy, trailing whitespace)
- Developer tooling (Makefile, .env.example)
- Initial ADRs for key architecture decisions
