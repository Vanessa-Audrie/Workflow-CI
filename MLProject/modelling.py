import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

df_train = pd.read_csv("Sleep_health_and_lifestyle_dataset_preprocessing/Sleep_health_and_lifestyle_dataset_train.csv")
df_test = pd.read_csv("Sleep_health_and_lifestyle_dataset_preprocessing/Sleep_health_and_lifestyle_dataset_test.csv")

X_train = df_train.drop(columns=['Sleep Disorder'])
y_train = df_train['Sleep Disorder']

X_test = df_test.drop(columns=['Sleep Disorder'])
y_test = df_test['Sleep Disorder']

print("\nX_train NaN:", X_train.isnull().sum().sum())
print("y_train NaN:", y_train.isnull().sum())
print("X_test NaN:", X_test.isnull().sum().sum())
print("y_test NaN:", y_test.isnull().sum())

with mlflow.start_run():
    mlflow.autolog()

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"\nAccuracy: {accuracy:.4f}")