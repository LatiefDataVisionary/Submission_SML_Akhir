import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub

from modelling_tuning import BASE_DIR

DAGSHUB_REPO_OWNER = "datasciencelatief"
DAGSHUB_REPO_NAME = "Submission_SML_Akhir"

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train_cleaned.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "test_cleaned.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts_baseline")

def load_dataset(train_path, test_path):
    """
    Memuat dataset latih dan uji dari direktori lokal.
    Memisahkan fitur independen (X) dan target dependen (y).
    """
    print("[INFO] Memuat dataset latih dan uji...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    return X_train, y_train, X_test, y_test

def train_baseline_model(X_train, y_train):
    """
    Melatih model Random Forest menggunakan parameter bawaan (baseline).
    Digunakan sebagai tolak ukur sebelum melakukan hyperparameter tuning.
    """
    print("[INFO] Melatih model baseline Random Forest...")
    
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi performa model menggunakan metrik klasifikasi standar.
    Mengembalikan dictionary berisi nilai metrik dan array hasil prediksi.
    """
    print("[INFO] Mengevaluasi performa model...")
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
    Membuat visualisasi Confusion Matrix dan menyimpannya sebagai file gambar.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Baseline Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    file_path = os.path.join(output_dir, 'confusion_matrix_baseline.png')
    plt.savefig(file_path)
    plt.close()
    
    return file_path

def generate_feature_importance(model, feature_names, output_dir):
    """
    Membuat visualisasi Feature Importance dan menyimpannya sebagai file gambar dan CSV.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances - Baseline Model")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    img_path = os.path.join(output_dir, 'feature_importance_baseline.png')
    plt.savefig(img_path)
    plt.close()
    
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    csv_path = os.path.join(output_dir, 'feature_importance_baseline.csv')
    importance_df.to_csv(csv_path, index=False)
    
    return img_path, csv_path

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    try:
        X_train, y_train, X_test, y_test = load_dataset(TRAIN_DATA_PATH, TEST_DATA_PATH)
    except FileNotFoundError:
        print("[ERROR] File dataset tidak ditemukan. Periksa kembali path direktori.")
        return

    print("[INFO] Menginisialisasi koneksi DagsHub...")
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    print("[INFO] Memulai MLflow run untuk model baseline...")
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        
        baseline_model = train_baseline_model(X_train, y_train)
        
        metrics, y_pred = evaluate_model(baseline_model, X_test, y_test)
        
        print("[INFO] Menghasilkan artefak evaluasi...")
        cm_path = generate_confusion_matrix(y_test, y_pred, ARTIFACT_DIR)
        fi_img_path, fi_csv_path = generate_feature_importance(baseline_model, X_train.columns, ARTIFACT_DIR)
        
        print("[INFO] Mencatat parameter, metrik, dan artefak ke MLflow...")
        
        model_params = baseline_model.get_params()
        mlflow.log_params(model_params)
        
        mlflow.log_metrics(metrics)
        
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(fi_img_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(fi_csv_path, artifact_path="evaluation_data")
        
        mlflow.sklearn.log_model(
            sk_model=baseline_model,
            name="model",
            registered_model_name="TelcoChurn_RandomForest_Baseline"
        )
        
        print("[INFO] Proses eksekusi baseline selesai dan berhasil dicatat di MLflow.")

if __name__ == "__main__":
    main()