import pandas as pd

# Cargar el CSV
df = pd.read_csv(r"C:\Users\maich\OneDrive\Documents\alzheimers_disease_data.csv")

# Mirar el target
if "Diagnosis" in df.columns:
    print("\n--- Target: Diagnosis ---")
    print(df["Diagnosis"].value_counts())

# 2. Análisis exploratorio de datos (EDA)

# Información general del DataFrame
print("\n--- Información del DataFrame ---")
print(df.info())
print(df.head())
print(df.describe(include='all'))

# Visualización de distribuciones
import matplotlib.pyplot as plt
import seaborn as sns
import math

n_cols = df.shape[1]   # número de columnas
n_rows = math.ceil(n_cols / 4)  # 4 histogramas por fila

fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4*n_rows))  
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(df[col], kde=True, color="purple", ax=axes[i])
    axes[i].set_title(f"{col}")
    
# Apagar los ejes vacíos (si sobran)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# - - - Matriz de correlación
import matplotlib.pyplot as plt
import seaborn as sns

# Elimino la columna 'DoctorInCharge' para calcular correlaciones
df_correlacion = df.drop(columns=["DoctorInCharge"])

plt.figure(figsize=(20,10))
sns.heatmap(df_correlacion.corr(), annot=True, cmap="magma" , fmt=".2f")
plt.title("Correlación entre variables")
plt.show()

# - - - Análisis bivariado
# Variables categóricas vs target
import matplotlib.pyplot as plt
import seaborn as sns
import math

cat_vars = ["Gender","Ethnicity","EducationLevel","Smoking","FamilyHistoryAlzheimers",
            "CardiovascularDisease","Diabetes","Depression","HeadInjury","Hypertension",
            "MemoryComplaints","BehavioralProblems","Confusion","Disorientation",
            "PersonalityChanges","DifficultyCompletingTasks","Forgetfulness"]

num_vars = ["Age","BMI","AlcoholConsumption","PhysicalActivity","DietQuality",
            "SleepQuality","SystolicBP","DiastolicBP","CholesterolTotal",
            "CholesterolLDL","CholesterolHDL","CholesterolTriglycerides",
            "MMSE","FunctionalAssessment","ADL"]

# --- Graficar categóricas en subplots ---
n_cols = len(cat_vars)
n_rows = math.ceil(n_cols / 4)   # 4 gráficas por fila

fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_vars):
    sns.countplot(x=col, hue="Diagnosis", data=df, palette="pastel", ax=axes[i])
    axes[i].set_title(f"{col} vs Diagnosis")

# Borrar ejes sobrantes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# --- Graficar numéricas en subplots ---
n_cols = len(num_vars)
n_rows = math.ceil(n_cols / 4)   # 4 gráficas por fila

fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_vars):
    sns.boxplot(x="Diagnosis", y=col, data=df, palette="Blues", ax=axes[i])
    axes[i].set_title(f"{col} vs Diagnosis")

# Borrar ejes sobrantes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#3. Preprocesamiento de datos

# Manejo de valores nulos
print("\n--- Valores nulos por columna ---")
print(df.isnull().sum())


# Separar ordinal y nominal ---
import pandas as pd

ordinal_vars = ["EducationLevel"]
nominal_vars = [col for col in cat_vars if col not in ordinal_vars]

# --- 2️⃣ Label Encoding para ordinal ---
orden_educ = {'Primaria':1, 'Secundaria':2, 'Universidad':3}
for col in ordinal_vars:
    df[col] = df[col].map(orden_educ)

# --- 3️⃣ One-Hot Encoding para nominal ---
df = pd.get_dummies(df, columns=nominal_vars, drop_first=True)  # drop_first=True evita multicolinealidad

# --- 4️⃣ DataFrame listo ---
print(df.head())
print("\nShape final:", df.shape)

# 4. División de datos y escalado
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

orden_educ = {'Primaria':1, 'Secundaria':2, 'Universidad':3}
df['EducationLevel'] = df['EducationLevel'].map(orden_educ)

y = df['Diagnosis']
X = df.drop(columns=['Diagnosis'])

num_vars = X.select_dtypes(include=['int64','float64']).columns
cat_vars = X.select_dtypes(include=['object','category']).columns

print("Numéricas:", list(num_vars))
print("Categóricas:", list(cat_vars))

pipeline = Pipeline([
    ('preprocess', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_vars),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ]), cat_vars)
    ])),
    ('pca', PCA(n_components=0.95))
])

# División 70/15/15 ---
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

X_train_proc = pipeline.fit_transform(X_train)
X_val_proc = pipeline.transform(X_val)
X_test_proc = pipeline.transform(X_test)

print("X_train shape:", X_train_proc.shape)
print("X_val shape:", X_val_proc.shape)
print("X_test shape:", X_test_proc.shape)

# 5. Modelado y evaluación

# - - - KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_proc, y_train)

print("Accuracy KNN:", knn.score(X_val_proc, y_val))

# - - - Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_proc, y_train)  # usar X_train_proc, no X_train crudo
y_pred_rf = rf.predict(X_test_proc)
print("Accuracy Random Forest:", accuracy_score(y_test, y_pred_rf))


# - - - Redes Neuronales
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

n_features = X_train_proc.shape[1]
n_classes = len(y.unique())  # detecta binario o multiclase

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid') if n_classes == 2 else Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_proc, y_train,
    validation_data=(X_val_proc, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
test_loss, test_acc = model.evaluate(X_test_proc, y_test, verbose=0)
print(f"Accuracy en DNN: {test_acc:.4f}")

# Graficar historial de entrenamiento y validación entre los tres modelos

import pandas as pd
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_proc, y_train)

rf_results = {
    "Train": accuracy_score(y_train, rf_model.predict(X_train_proc)),
    "Val": accuracy_score(y_val, rf_model.predict(X_val_proc)),
    "Test": accuracy_score(y_test, rf_model.predict(X_test_proc))
}

dnn_train_acc = model.evaluate(X_train_proc, y_train, verbose=0)[1]
dnn_val_acc   = model.evaluate(X_val_proc, y_val, verbose=0)[1]
dnn_test_acc  = model.evaluate(X_test_proc, y_test, verbose=0)[1]

dnn_results = {
    "Train": dnn_train_acc,
    "Val": dnn_val_acc,
    "Test": dnn_test_acc
}

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_proc, y_train)

knn_results = {
    "Train": accuracy_score(y_train, knn_model.predict(X_train_proc)),
    "Val": accuracy_score(y_val, knn_model.predict(X_val_proc)),
    "Test": accuracy_score(y_test, knn_model.predict(X_test_proc)) }

results_df = pd.DataFrame({
    "RandomForest": rf_results,
    "DNN": dnn_results,
    "kNN": knn_results
}).T  # transponer para que cada modelo sea fila

print("\n📊 Resultados comparativos:")
print(results_df)

import matplotlib.pyplot as plt

results_df.plot(kind="bar", figsize=(8,5))
plt.title("Desempeño de modelos")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.show()

# 6.Prueba artificial

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


# Quitamos columnas que no aportan al modelo
features = df.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"]).columns.tolist()

X = df[features].copy()
y = df["Diagnosis"]

# Codificar Diagnosis si es categórico
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# Escalar variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

n_features = X_train.shape[1]
n_classes = len(np.unique(y))

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid') if n_classes == 2 else Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy' if n_classes == 2 else 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy en test: {test_acc:.4f}")
