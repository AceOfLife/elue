# GRP Prediction Model using HalvingRandomSearchCV

This repository contains a two-stage machine learning pipeline:
- **Stage 1:** Predicts log_TRP from media spend and context
- **Stage 2:** Predicts log_GRP from predicted TRP and additional features

## Usage
You can load the saved pipelines (`rf_pipeline.pkl` and `rf_pipeline2.pkl`) and apply them to new data using `joblib`.

## License
Licensed under the **Apache License 2.0**.  
See [LICENSE](./LICENSE) for full terms.
