import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import os

# =====================================
# 1. SET EXPERIMENT (TANPA tracking_uri)
# =====================================
mlflow.set_experiment("NASA Asteroid Scoring")

# =====================================
# 2. PATH DATA (AMAN)
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

X_path = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "nearest_earth_object_preprocessing",
    "X_train.csv"
)

y_path = os.path.join(
    PROJECT_ROOT,
    "preprocessing",
    "nearest_earth_object_preprocessing",
    "y_train.csv"
)

print("X exists:", os.path.exists(X_path))
print("Y exists:", os.path.exists(y_path))

X_train = pd.read_csv(X_path)
y_train = pd.read_csv(y_path)

# =====================================
# 3. AUTOLOG
# =====================================
mlflow.sklearn.autolog()

# =====================================
# 4. TRAINING
# =====================================
with mlflow.start_run(run_name="NASA_Asteroid_Basic"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train.values.ravel())

    print("✅ Modelling Basic selesai!")
