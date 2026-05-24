import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub

DAGSHUB_REPO_OWNER = "datasciencelatief"
DAGSHUB_REPO_NAME = "Submission_SML_Akhir"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train_cleaned.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test_cleaned.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

def load_dataset(train_path, test_path):
    """
    Memuat dataset latih dan uji dari direktori lokal.
    """
    print(f"[INFO] Memuat dataset dari {train_path}...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {train_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    return X_train, y_train, X_test, y_test

def tune_hyperparameters(X_train, y_train):
    """
    Melakukan hyperparameter tuning menggunakan GridSearchCV.
    """
    print("[INFO] Memulai hyperparameter tuning...")
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"[INFO] Tuning selesai. Parameter terbaik: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi performa model.
    """
    print("[INFO] Mengevaluasi model pada data uji...")
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics, y_pred

def generate_confusion_matrix(y_test, y_pred, output_dir):
    """
    Menghasilkan plot Confusion Matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Tuning Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    file_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(file_path)
    plt.close()
    return file_path

def generate_feature_importance(model, feature_names, output_dir):
    """
    Menghasilkan plot Feature Importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances - Tuning Model")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    img_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(img_path)
    plt.close()
    
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(csv_path, index=False)
    
    return img_path, csv_path

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # 1. AUTHENTICATION FIX: Mencegah OAuth interaktif di GitHub Actions
    print("[INFO] Menginisialisasi koneksi DagsHub...")
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["DAGSHUB_USER_TOKEN"] = token
        print("[INFO] Token ditemukan, menggunakan Headless Authentication.")
    
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    # Set Tracking URI secara eksplisit
    tracking_uri = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    try:
        X_train, y_train, X_test, y_test = load_dataset(TRAIN_DATA_PATH, TEST_DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal memuat data: {e}")
        return

    print("[INFO] Memulai MLflow run...")
    with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):
        
        # Latih model dengan tuning
        best_model, best_params = tune_hyperparameters(X_train, y_train)
        
        # Evaluasi
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        
        # Artefak
        cm_path = generate_confusion_matrix(y_test, y_pred, ARTIFACT_DIR)
        fi_img_path, fi_csv_path = generate_feature_importance(best_model, X_train.columns, ARTIFACT_DIR)
        
        # MANUAL LOGGING (Syarat Skilled)
        print("[INFO] Mencatat parameter dan metrik secara manual...")
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(fi_img_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(fi_csv_path, artifact_path="evaluation_data")
        
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="TelcoChurn_RandomForest_Tuning"
        )
        
        print("[INFO] Eksperimen tuning berhasil dicatat di DagsHub/MLflow.")

if __name__ == "__main__":
    main()
