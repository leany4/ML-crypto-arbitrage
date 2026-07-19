```text
models/
├── q35_perp/
│   ├── model.joblib
│   └── manifest.json
├── transformer/
│   ├── multiscale_transformer_best.pt
│   ├── feature_scaler.npz
│   ├── category_config.json
│   ├── model_config.json
│   ├── dataset_contract.json
│   └── manifest.json
└── rl_agent/
    ├── lstm_r2d2.pt
    ├── feature_scaler.joblib
    ├── evaluation_summary.json
    └── manifest.json
```

Скрипты для упаковки manifest файлов:

```text
scripts/package_q35_bundle.py
scripts/package_transformer_bundle.py
scripts/package_rl_r2d2_bundle.py
```