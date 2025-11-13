import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_data():
    train = pd.read_csv("data/train.csv")

    # Log-transform the target
    train["SalePrice_log"] = np.log1p(train["SalePrice"])
    return train


# ----------------------------------------------------------------------
# 2. Missing value handling
# ----------------------------------------------------------------------
def fill_missing_values(df):
    # A) Categorical "None"
    no_feature_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in no_feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # B) Numeric zero
    zero_fill_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath'
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # C) Neighborhood-based median LotFrontage
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    # D) Remaining missing values → most frequent or 0
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(0)

    return df


# ----------------------------------------------------------------------
# 3. Build preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocessor(df):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols and c not in ["SalePrice", "SalePrice_log"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    return preprocessor, cat_cols, num_cols


# ----------------------------------------------------------------------
# 4. Train XGBoost model (optimized)
# ----------------------------------------------------------------------
def train_model(preprocessor, X_train, y_train):
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():

    print("Loading data...")
    df = load_data()

    print("Cleaning missing values...")
    df = fill_missing_values(df)

    print("Splitting data...")
    X = df.drop(["SalePrice", "SalePrice_log"], axis=1)
    y = df["SalePrice_log"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building preprocessor...")
    preprocessor, _, _ = build_preprocessor(df)

    print("Training model...")
    model = train_model(preprocessor, X_train, y_train)

    print("Evaluating model...")
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    print(f"Validation RMSE: {rmse:.4f}")

    print("Saving model...")
    with open("model/model_xgb_optimized.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
