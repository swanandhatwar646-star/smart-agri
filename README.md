# Smart Agriculture Prediction System

## Overview
A comprehensive Streamlit application for crop recommendation, irrigation planning, and yield forecasting using machine learning models.

## Features
- **Crop Recommendation**: Recommends optimal crops based on soil nutrients and weather conditions
- **Irrigation Recommendation**: Predicts irrigation needs based on field conditions
- **Yield Prediction**: Forecasts crop yield using historical data

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Project Structure
```
crop-prediction-deploy/
├── app.py                 # Main Streamlit application
├── data/                  # Dataset files
│   ├── Crop_Recommendation.csv
│   ├── indian crop production.csv
│   └── irrigation_recommendation_dataset.csv
├── models/                # Trained ML models
│   ├── crop_recommendation_model.pkl
│   ├── irrigation_model.pkl
│   └── yield_prediction_model.pkl
├── requirements.txt        # Python dependencies
└── README.md             # This file
```

## Usage
1. Run `streamlit run app.py`
2. Open browser to `http://localhost:8501`
3. Use sidebar navigation to switch between features:
   - Crop Recommendation
   - Irrigation Recommendation  
   - Yield Prediction

## Model Information
- **Crop Model**: Random Forest (99.3% accuracy)
- **Irrigation Model**: Random Forest (100% accuracy)
- **Yield Model**: CatBoost Regressor (R² = 0.891)

## Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Libraries**: scikit-learn, pandas, numpy
- **Model Persistence**: joblib
