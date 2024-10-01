# Lengua_Manos
Mega proyecto de reconocimiento de gestos mediante entrenamiento por redes neuronales de lengas de señas.

Acceder al entorno virtual

python -m venv env
env\Scripts\activate
pip install -r requirements.txt

Primero se capturan los datos utilizando

python src/capture_data.py

Despues se ejecuta el reconocimiento de gestos

python src/gesture_recognition.py