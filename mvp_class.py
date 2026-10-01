import os
import json
import hashlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

import wandb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import random
from datetime import datetime

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = f"Car_Price_Prediction_MVP"

class Car_Training_Model():
    def __init__(self):
        self.data_path = Path("uc-bana-7075/MVP/National_data_2_locations_Clean.csv") # Update this
        self.random_state = random.randint(0, 2**8)

        random.seed(self.random_state) # Added, you were not actually tracking randomc before.
        np.random.seed(self.random_state)

        print(f"Random state: {self.random_state}") 
        
        self.artifact_dir = Path("artifacts")
        self.model_dir = Path("models")
        self.processed_dir = Path("data")
    
        self.training_sample_size = 50000 # Why?
    
        self.desired_predictor = "Selling Price"
    
        self.artifact_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        self.ingest_data()
    
        print(f"Data Path: {self.data_path}")
        print(f"Desired Prediction Target: {self.desired_predictor}")
        print(f"Sample Size: {self.training_sample_size}")

    def ingest_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Could not find {self.data_path}. "
                "Place the cleaned CSV in the same directory as this notebook "
                "or update self.data_path in the configuration cell."
            )

        self.df = pd.read_csv(self.data_path)

        print("Batch ingestion complete.")
        print(f"Rows: {len(self.df):,}")
        print(f"Columns: {self.df.shape[1]}")
        print(f"File size: {self.data_path.stat().st_size / (1024**2):.2f} MB")

        print(self.df.head())

        self.data_hash = self.sha256_file()

        self.data_version = {
            "source_file": self.data_path.name,
            "sha256": self.data_hash,
            "rows": int(len(self.df)),
            "columns": int(self.df.shape[1]),
            "target": self.desired_predictor,
            "random_state": self.random_state,
            "mvp_sample_size": self.training_sample_size,
        }

        with open(self.artifact_dir / "data_version.json", "w") as f:
            json.dump(self.data_version, f, indent=4)

        print("Data version fingerprint:")
        print(self.data_hash)
        print("\nMetadata saved to:", self.artifact_dir / "data_version.json")

    def sha256_file(self, chunk_size = 1024 * 1024):
        self.sha = hashlib.sha256()
        with open(self.data_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                self.sha.update(chunk)
        return self.sha.hexdigest()

    def data_validation(self):
        self.required_columns = [
            "Year", "Make", "Model", "Trim", "Body", "Transmission",
            "Vehicle Identification Number", "State Code", "Odometer",
            "Color", "Interior", "Seller", "Market Price", "Selling Price",
            "Sale Date", "Sale Year", "Sale Month", "Sale Month Name",
            "Sale Day of Week", "Sale Day Name", "State"
        ]

        self.validation = {
            "required_columns_present": set(self.required_columns).issubset(self.df.columns),
            "missing_values": int(self.df.isna().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "negative_selling_prices": int((self.df[self.desired_predictor] < 0).sum()),
            "negative_odometer": int((self.df["Odometer"] < 0).sum()),
            "invalid_vehicle_years": int(((self.df["Year"] < 1900) | (self.df["Year"] > 2026)).sum()),
            "non_numeric_target": not pd.api.types.is_numeric_dtype(self.df[self.desired_predictor]),
        }

        print("Validation results:")
        for check, result in self.validation.items():
            print(f"  {check}: {result}")

        # Hard-fail only on conditions that make model training unsafe.
        hard_fail_checks = {
            "required_columns_present": self.validation["required_columns_present"],
            "no_missing_values": self.validation["missing_values"] == 0,
            "non_negative_target": self.validation["negative_selling_prices"] == 0,
            "non_negative_odometer": self.validation["negative_odometer"] == 0,
            "numeric_target": self.validation["non_numeric_target"],
        }

        # Note: the 'numeric_target' check above is intentionally named for readability.
        # A numeric target should make this True; therefore check the dtype directly.
        hard_fail_checks["numeric_target"] = pd.api.types.is_numeric_dtype(self.df[self.desired_predictor])

        failed = [name for name, passed in hard_fail_checks.items() if not passed]

        if failed:
            raise ValueError(f"Data validation failed: {failed}")

        with open(self.artifact_dir / "validation_report.json", "w") as f:
            json.dump(self.validation, f, indent=4)

        print("\nAll required validation checks passed.")

    def features(self):
        self.model_df = self.df.copy()
        self.model_df["Vehicle Age"] = self.model_df["Sale Year"] - self.model_df["Year"]

        self.numeric_features = [
            "Year",
            "Odometer",
            "Market Price",
            "Sale Year",
            "Sale Month",
            "Sale Day of Week",
            "Vehicle Age",
        ]

        self.categorical_features = [
            "Make",
            "Model",
            "Trim",
            "Body",
            "Transmission",
            "Color",
            "Interior",
            "State",
        ]

        self.feature_columns = self.numeric_features + self.categorical_features

        self.X = self.model_df[self.feature_columns]
        self.y = self.model_df[self.desired_predictor]

        print("Feature matrix shape:", self.X.shape)
        print("Target shape:", self.y.shape)
        print("\nNumeric features:", self.numeric_features)
        print("Categorical features:", self.categorical_features)

        # Store feature documentation for reproducibility.
        self.feature_info = {
            "target": self.desired_predictor,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "excluded_columns": [
                "Vehicle Identification Number",
                "Sale Date",
                "Sale Month Name",
                "Sale Day Name",
                "State Code",
                "Seller"
            ],
        }

        with open(self.artifact_dir / "feature_info.json", "w") as f:
            json.dump(self.feature_info, f, indent=4)

    def split_data(self):
        self.sample_n = min(self.training_sample_size, len(self.model_df))

        sample_df = self.model_df.sample(
            n=self.sample_n,
            random_state=self.random_state
        ).reset_index(drop=True)

        X_sample = sample_df[self.feature_columns]
        y_sample = sample_df[self.desired_predictor]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_sample,
            y_sample,
            test_size=0.20,
            random_state=self.random_state
        )

        print(f"Sample rows: {len(sample_df):,}")
        print(f"Training rows: {len(self.X_train):,}")
        print(f"Testing rows: {len(self.X_test):,}")

        # Save the exact modeling sample so another group member can reproduce the demo.
        sample_df.to_parquet(
            self.processed_dir / "mvp_modeling_sample.parquet",
            index=False
        )

        print("\nProcessed MVP dataset saved to:")
        print(self.processed_dir / "mvp_modeling_sample.parquet")       

    def preprocessing(self):
        self.preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                self.numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2
                ),
                self.categorical_features
            ),
        ],
        remainder="drop"
    )

    def make_pipeline(self, model):
        """Create a complete preprocessing + model pipeline."""
        return Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("model", model),
            ]
        )

    def evaluate_model(self, model, X_test, y_test):
        """Calculate common regression metrics."""
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

    def ridge(self, iteration, alpha = 10.0):
        self.ridge_model = self.make_pipeline(Ridge(alpha))

        ridge_params = {
            "model_type": "Ridge Regression",
            "alpha": alpha,
            "train_rows": len(self.X_train),
            "test_rows": len(self.X_test),
            "data_sha256": self.data_hash,
            "random_state": self.random_state,
        }

        with wandb.init(project = EXPERIMENT_NAME, group = "Ridge", name = f"Ridge_{iteration}", config = {**ridge_params}) as run:
            
            self.ridge_model.fit(self.X_train, self.y_train)
            self.ridge_metrics = self.evaluate_model(self.ridge_model, self.X_test, self.y_test)

            # for metric_name, metric_value in self.ridge_metrics.items():
            #     mlflow.log_metric(metric_name, metric_value)

            wandb.log(self.ridge_metrics)

            artifact = wandb.Artifact(name=f"Ridge_artifacts_{iteration}", type = "dataset_and_reports")
            artifact.add_file(str(self.artifact_dir / "data_version.json"))
            artifact.add_file(str(self.artifact_dir / "validation_report.json"))
            artifact.add_file(str(self.artifact_dir / "feature_info.json"))
            run.log_artifact(artifact)

            model_path = "model.joblib"
            joblib.dump(self.ridge_model, model_path)
            model_artifact = wandb.Artifact(name = f"Ridge_model_{iteration}", type = "model")
            model_artifact.add_file(model_path)
            run.log_artifact(model_artifact)
            # mlflow.sklearn.log_model(
            #     self.rf_model,
            #     "model",
            #     skops_trusted_types=[
            #         "sklearn.tree._tree.Tree"
            #     ]
            # )

            self.ridge_run_id = run.id

    def lasso(self, iteration,  alpha = 0.001, max_iter = 5000):
        self.lasso_model = self.make_pipeline(
            Lasso(
                alpha=alpha,
                max_iter=max_iter,
                selection="random",
                random_state=self.random_state
            )
        )

        lasso_parms = {
            "model_type": "Lasso Regression",
            "alpha": alpha,
            "max_iter": max_iter,
            "train_rows": len(self.X_train),
            "test_rows": len(self.X_test),
            "data_sha256": self.data_hash,
            "random_state": self.random_state
        }

        with wandb.init(project = EXPERIMENT_NAME, group = "Lasso", name = f"Lasso_{iteration}", config = {**lasso_parms}) as run:
            self.lasso_model.fit(self.X_train, self.y_train)
            self.lasso_metrics = self.evaluate_model(self.lasso_model, self.X_test, self.y_test)

            # mlflow.log_param("model_type", "Lasso Regression")
            # mlflow.log_param("alpha", alpha)
            # mlflow.log_param("max_iter", max_iter)
            # mlflow.log_param("train_rows", len(self.X_train))
            # mlflow.log_param("test_rows", len(self.X_test))
            # mlflow.log_param("data_sha256", self.data_hash)
            # mlflow.log_param("random_state", self.random_state)
            wandb.log(self.lasso_metrics)

            # --------------------------------------------------------
            # Log supporting artifacts
            # --------------------------------------------------------
            artifact = wandb.Artifact(name=f"Lasso_artifacts_{iteration}", type = "dataset_and_reports")
            artifact.add_file(str(self.artifact_dir / "data_version.json"))
            artifact.add_file(str(self.artifact_dir / "validation_report.json"))
            artifact.add_file(str(self.artifact_dir / "feature_info.json"))
            run.log_artifact(artifact)

            # --------------------------------------------------------
            # MLflow model logging
            #
            # Random Forest contains sklearn.tree._tree.Tree objects.
            # We explicitly trust this exact type because the model was
            # trained locally from our own trusted source code/data.
            # --------------------------------------------------------
            model_path = "model.joblib"
            joblib.dump(self.lasso_model, model_path)
            model_artifact = wandb.Artifact(name = f"Lasso_model_{iteration}", type = "model")
            model_artifact.add_file(model_path)
            run.log_artifact(model_artifact)
            # mlflow.sklearn.log_model(
            #     self.rf_model,
            #     "model",
            #     skops_trusted_types=[
            #         "sklearn.tree._tree.Tree"
            #     ]
            # )

            self.lasso_run_id = run.id

        print("Lasso metrics:", self.lasso_metrics)
        print("MLflow run ID:", self.lasso_run_id)

    def random_forest(self, iteration, n_estimators = 500):
        rf_params = {
            "n_estimators": n_estimators,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": 0.5,
            "n_jobs": -1,
            "random_state": self.random_state,
        }

        self.rf_model = self.make_pipeline(
            RandomForestRegressor(**rf_params)
        )

        with wandb.init(project = EXPERIMENT_NAME, group = "Random_Forest", name = f"Random_Forest_{iteration}", config = {
            "model_type": "Random Forest",
            **rf_params,
            "train_rows": len(self.X_train),
            "test_rows": len(self.y_test),
            "data_sha256": self.data_hash,
        }) as run:
            
            self.rf_model.fit(self.X_train, self.y_train)

            self.rf_metrics = self.evaluate_model(
                self.rf_model,
                self.X_test,
                self.y_test
            )

            # Log model parameters
            # mlflow.log_params({
            #     "model_type": "Random Forest",
            #     **rf_params,
            #     "train_rows": len(self.X_train),
            #     "test_rows": len(self.y_test),
            #     "data_sha256": self.data_hash,
            # })

            # Log evaluation metrics
            wandb.log(self.rf_metrics)
            
            # --------------------------------------------------------
            # Log supporting artifacts
            # --------------------------------------------------------
            artifact = wandb.Artifact(name=f"RF_artifacts_{iteration}", type = "dataset_and_reports")
            artifact.add_file(str(self.artifact_dir / "data_version.json"))
            artifact.add_file(str(self.artifact_dir / "validation_report.json"))
            artifact.add_file(str(self.artifact_dir / "feature_info.json"))
            run.log_artifact(artifact)

            # --------------------------------------------------------
            # MLflow model logging
            #
            # Random Forest contains sklearn.tree._tree.Tree objects.
            # We explicitly trust this exact type because the model was
            # trained locally from our own trusted source code/data.
            # --------------------------------------------------------
            model_path = "model.joblib"
            joblib.dump(self.rf_model, model_path)
            model_artifact = wandb.Artifact(name = f"RF_model_{iteration}", type = "model")
            model_artifact.add_file(model_path)
            run.log_artifact(model_artifact)
            # mlflow.sklearn.log_model(
            #     self.rf_model,
            #     "model",
            #     skops_trusted_types=[
            #         "sklearn.tree._tree.Tree"
            #     ]
            # )

            self.rf_run_id = run.id

        print("Random Forest metrics:", self.rf_metrics)
        print("MLflow run ID:", self.rf_run_id)

    def xgBoost(self, iteration, n_estimators = 300, max_depth = 8, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8):
        xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": self.random_state,
        }

        self.xgb_model = self.make_pipeline(
            XGBRegressor(**xgb_params)
        )

        # with mlflow.start_run(run_name=f"XGBoost_Ensemble_{iteration}") as run:\
        with wandb.init(project = EXPERIMENT_NAME, group = "XGBoost", name = f"XGBoost_Ensemble_{iteration}", config = {
            "model_type": "XGBoost",
            **xgb_params,
            "train_rows": len(self.X_train),
            "test_rows": len(self.X_test),
            "data_sha256": self.data_hash,
        }) as run:

            # --------------------------------------------------------
            # Train model
            # --------------------------------------------------------
            self.xgb_model.fit(self.X_train, self.y_train)

            # --------------------------------------------------------
            # Evaluate model
            # --------------------------------------------------------
            self.xgb_metrics = self.evaluate_model(self.xgb_model, self.X_test, self.y_test)

            # --------------------------------------------------------
            # Log hyperparameters
            # --------------------------------------------------------
            # mlflow.log_params({
            #     "model_type": "XGBoost",
            #     **xgb_params,
            #     "train_rows": len(self.X_train),
            #     "test_rows": len(self.X_test),
            #     "data_sha256": self.data_hash,
            # })

            # --------------------------------------------------------
            # Log evaluation metrics
            # --------------------------------------------------------
            # for metric_name, metric_value in self.xgb_metrics.items():
            #     mlflow.log_metric(
            #         metric_name,
            #         metric_value
            #     )
            wandb.log(self.xgb_metrics)

            # --------------------------------------------------------
            # Log supporting artifacts
            # --------------------------------------------------------
            artifact = wandb.Artifact(name=f"XGBoost_artifacts_{iteration}", type = "dataset_and_reports")
            artifact.add_file(str(self.artifact_dir / "data_version.json"))
            artifact.add_file(str(self.artifact_dir / "validation_report.json"))
            artifact.add_file(str(self.artifact_dir / "feature_info.json"))
            run.log_artifact(artifact)
            # mlflow.log_artifact(
            #     str(self.artifact_dir / "data_version.json")
            # )

            # mlflow.log_artifact(
            #     str(self.artifact_dir / "validation_report.json")
            # )

            # mlflow.log_artifact(
            #     str(self.artifact_dir / "feature_info.json")
            # )

            # --------------------------------------------------------
            # Log XGBoost model to MLflow
            #
            # MLflow/skops identifies these XGBoost classes as
            # untrusted during serialization:
            #
            #   xgboost.core.Booster
            #   xgboost.sklearn.XGBRegressor
            #
            # We explicitly trust ONLY those two types.
            # --------------------------------------------------------
            model_path = "model.joblib"
            joblib.dump(self.xgb_model, model_path)
            model_artifact = wandb.Artifact(name = f"XGBoost_model_{iteration}", type = "model")
            model_artifact.add_file(model_path)
            run.log_artifact(model_artifact)
            # mlflow.sklearn.log_model(
            #     self.xgb_model,
            #     "model",
            #     skops_trusted_types=[
            #         "xgboost.core.Booster",
            #         "xgboost.sklearn.XGBRegressor"
            #     ]
            # )

            self.xgb_run_id = run.id

        print("XGBoost metrics:", self.xgb_metrics)
        print("Run ID:", self.xgb_run_id)

    def full_model_train(self, iteration):
        self.ridge(iteration)
        self.lasso(iteration)
        self.random_forest(iteration)
        self.xgBoost(iteration)

    def model_comparison(self):

        results = pd.DataFrame([
            {"Model": "Ridge Regression", **self.ridge_metrics, "Run ID": self.ridge_run_id},
            {"Model": "Lasso Regression", **self.lasso_metrics, "Run ID": self.lasso_run_id},
            {"Model": "Random Forest", **self.rf_metrics, "Run ID": self.rf_run_id},
            {"Model": "XGBoost", **self.xgb_metrics, "Run ID": self.xgb_run_id},
        ])

        results = results.sort_values("RMSE").reset_index(drop=True) # Why RMSE?

        results.to_csv(self.artifact_dir / "model_comparison.csv", index=False)

        print("Comparison saved to:", self.artifact_dir / "model_comparison.csv")

        model_objects = {
            "Ridge Regression": self.ridge_model,
            "Lasso Regression": self.lasso_model,
            "Random Forest": self.rf_model,
            "XGBoost": self.xgb_model,
        }

        best_model_name = results.iloc[0]["Model"]
        best_model = model_objects[best_model_name]
        best_run_id = results.iloc[0]["Run ID"]

        best_model_path = self.model_dir / "best_car_price_model.joblib"
        joblib.dump(best_model, best_model_path)

        print(f"Selected MVP model based on test RMSE: {best_model_name}")
        print(f"Run ID: {best_run_id}")
        print(f"Saved model: {best_model_path}")
        return best_run_id

def main():
    # RANDOM_STATE = 42
    print("Dependencies loaded successfully.")

    # mlflow_db = "sqlite:///mlflow.db"
    # mlflow.set_tracking_uri(mlflow_db)

    print("MLflow tracking URI:", mlflow.get_tracking_uri()) 
        
    # mlflow.set_experiment(EXPERIMENT_NAME)
    print("Experiment:", EXPERIMENT_NAME)
    
    num_iterations = 20
    for iteration in range(0, num_iterations):
        new_model_round = Car_Training_Model()
        new_model_round.data_validation()
        new_model_round.features()
        new_model_round.split_data()
        new_model_round.preprocessing()     

        new_model_round.full_model_train(iteration)

        best_run_id = new_model_round.model_comparison()

        MODEL_REGISTRY_NAME = "CarPricePrediction"

        client = mlflow.tracking.MlflowClient()

        # The best model was already logged under its original run.
        # Register that logged model as a version in the model registry.
        model_uri = f"runs:/{best_run_id}/model"

        try:
            registered = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_REGISTRY_NAME
            )

            version_number = registered.version

            # MLflow aliases provide a current-model pointer.
            client.set_registered_model_alias(
                MODEL_REGISTRY_NAME,
                "champion",
                version_number
            )

            print(f"Registered model: {MODEL_REGISTRY_NAME}")
            print(f"Version: {version_number}")
            print("Alias: champion")

        except Exception as e:
            print("Model registry step did not complete.")
            print("If your MLflow version/backend does not support local registry storage,")
            print("the model is still versioned by its MLflow run ID and saved artifact.")
            print("Details:", e)

# This should do everything up to 18
if __name__ == "__main__":
    main()