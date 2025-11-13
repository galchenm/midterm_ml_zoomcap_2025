# House Price Prediction – ML Zoomcamp Midterm Project

This project implements an **end-to-end machine learning system** that predicts house sale prices using the **House Prices: Advanced Regression Techniques** dataset from Kaggle.
The pipeline includes:

* Data preprocessing & EDA
* Feature engineering using scikit-learn
* Training multiple ML models including XGBoost
* Hyperparameter tuning
* Saving a full inference pipeline
* Deploying a FastAPI service inside Docker
* Serving predictions via REST API

---

# 📊 1. Problem Description

The task is to predict the **SalePrice** of a house using 80+ categorical and numerical features such as:

* Overall house quality
* Living area size
* Lot size
* Garage capacity
* Neighborhood
* Year built and renovated
* Basement and exterior attributes

This is a supervised regression problem.
The target variable (`SalePrice`) is **log-transformed** during training to reduce skew.

---

# 📁 2. Dataset

Dataset:
[https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Files used:

* `train.csv` (training & validation)
* `test.csv` (optional for comparison to Kaggle leaderboard)

---

# 🔍 3. Exploratory Data Analysis (EDA)

Key steps performed:

* Distribution plots for numerical features
* Correlation matrix to identify important predictors
* Boxplots of SalePrice vs. top categorical features
* Analysis of missing values

### Key findings:

* SalePrice is right-skewed → log1p transformation applied
* Missing values in categorical features represent actual categories ("No Garage", "No Basement") → filled with `"missing"` string
* Strong predictors: `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`
* Some numerical columns are actually categorical (e.g., MSSubClass)

---

# 🛠 4. Data Preparation

Performed using **scikit-learn ColumnTransformer**:

### Numerical features

* Missing values → median
* No scaling (tree models do not require normalization)

### Categorical features

* Missing values → `"missing"`
* Encoded using `OrdinalEncoder`

A combined **training pipeline** ensures preprocessing is identical during inference.

---

# 🤖 5. Model Training

Three models were evaluated:

1. **Decision Tree Regressor**
2. **Random Forest Regressor**
3. **XGBoost Regressor**

### Evaluation Metric

RMSE on validation set using log-transformed targets.

### Selected Model

⭐ **XGBoost Regressor (with tuned hyperparameters)**

Achieved:

```
Validation RMSE ≈ 0.0104
```

XGBoost consistently outperformed baseline tree models.

---

# 💾 6. Saving the Model

A complete prediction pipeline was saved using `pickle`:

```
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", XGBRegressor(...))
    ]
)
```

Saved to:

```
model/model_xgb_optimized.pkl
```

This file is loaded directly by the FastAPI service.

---

# 🌐 7. FastAPI Web Service

Located in:

```
app/app.py
```

### Endpoints

| Method | Endpoint   | Description                  |
| ------ | ---------- | ---------------------------- |
| GET    | `/`        | Healthcheck                  |
| POST   | `/predict` | Returns predicted sale price |

### Automatic Feature Handling

If the input JSON does **not** include all features:

* Missing categorical → `"missing"`
* Missing numerical → `NaN`

Thus, even incomplete inputs are accepted.

---

# 🐳 8. Docker Deployment

### Build the Docker image

```bash
docker build -t house-price-api .
```

### Run the container

```bash
docker run -p 8000:8000 house-price-api
```

API will be available at:

```
http://localhost:8000
```

Interactive docs:

```
http://localhost:8000/docs
```

---

# 🧪 9. Test the API with curl (Working Example)

Use this tested and validated example:

### ✅ Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MSSubClass": 20,
    "MSZoning": "RL",
    "LotFrontage": 70,
    "LotArea": 7000,
    "Street": "Pave",
    "OverallQual": 6,
    "OverallCond": 5,
    "YearBuilt": 1990,
    "GrLivArea": 1500,
    "FullBath": 2,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 7,
    "GarageCars": 2,
    "GarageArea": 500
  }'
```

### 🟢 Example Response

```json
{
  "predicted_price": 183271.53125,
  "predicted_price_rounded": 183271.53
}
```

This confirms the model is loaded correctly and predictions are functioning.

---

# 📦 10. Project Structure

```
.
├── app/
│   └── app.py
├── model/
│   └── model_xgb_optimized.pkl
├── scripts/
│   └── train.py
├── notebooks/
│   └── notebook.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🎯 Final Notes

This project demonstrates a complete ML system:

* From data exploration
* To model training
* To production-ready API
* To containerized deployment
