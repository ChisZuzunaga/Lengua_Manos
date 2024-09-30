import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar los datos desde el archivo CSV
data = pd.read_csv('lengua_manos/data/gestures_data.csv')

# Separar características (landmarks) de las etiquetas
X = data.iloc[:, :-1].values  # Landmarks
y = data.iloc[:, -1].values   # Etiquetas (letras/gestos)

# Codificar las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
model.save('lengua_manos/models/modelo_gestos.h5')

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Precisión del modelo: {test_acc * 100:.2f}%')
