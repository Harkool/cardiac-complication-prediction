# 🫀 Cardiac Surgery Postoperative Complication Prediction

This project aims to develop machine learning models to predict pulmonary complications after cardiac surgery, and evaluate the causal effect of nerve block analgesia.

## 🔧 Features

- Predicts 7-day postoperative pulmonary complications
- Compares ML algorithms: Logistic Regression, XGBoost, DNN
- Explains predictions using SHAP
- Evaluates intervention effect using causal inference (T-Learner)
- Supports multi-task learning and contrastive learning modules

## 📁 Project Structure

- `data/`: raw and processed datasets
- `notebooks/`: exploratory analysis and model training
- `src/`: reusable functions and models
- `scripts/`: runnable training/analysis scripts
- `outputs/`: model results and visualizations

## ▶️ Quick Start

```bash
pip install -r requirements.txt
python scripts/train_model.py
```

## 📊 Example Output

Outputs will be stored in `outputs/` directory after training.

## 📄 License

MIT License
