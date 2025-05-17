# Heart Disease ML Models Script

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv("original data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "SVM": SVC(kernel='rbf', probability=True),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Linear Regression": LinearRegression()
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if name == "Linear Regression":
        y_pred = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    pd.DataFrame(y_pred).to_csv(f"Results/predictions_{name.replace(' ', '')}_model.csv", index=False)

# ANN - To be run in Google Colab due to TensorFlow requirements
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# model = Sequential()
# model.add(Dense(64, input_dim=13, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=100, batch_size=8)
# y_pred_ann = (model.predict(X_test) > 0.5).astype(int)
# pd.DataFrame(y_pred_ann).to_csv("Results/predictions_ANN_model.csv", index=False)

# Print results
for name, acc in results.items():
    print(f"{name} Accuracy: {acc:.4f}")
