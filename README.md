Para hacer funcionar el código debes tener anaconda instalado.

Esta es la estructura que debe tener la carpeta que almacenará el proyecto
/sign_language_translator
│
├── /data                     # Almacena las imágenes de los gestos y letras
│   ├── /letra_A               # Carpeta para la letra A
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   ├── /letra_B               # Carpeta para la letra B
│   │   ├── img1.png
│   │   └── ...
│   └── ...                    # Otras letras y gestos
│
├── /src                      # Código fuente
│   ├── capture_data.py        # Captura de datos de gestos/letras
│   ├── train_model.py         # Entrenamiento del modelo
│   └── gesture_recognition.py # Reconocimiento de gestos/letras
│
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Descripción del proyecto

Abre una terminal de anaconda con modo administrador para posterior añadir todas las dependencias necesarias para el correcto funcionamiento del código.

1.- Crear un entorno virtual de anaconda
    conda create --name sign_language_env python=3.8
    
2.- Activar el entorno virtual
    conda activate sign_language_env

3.- Instalar dependencias
    conda install tensorflow
    conda install tensorflow-gpu
    conda install opencv
    pip install mediapipe
    conda install numpy

Luego se va a la ruta del proyecto utilizando "cd C:\Users\queso\OneDrive\Escritorio\Lengu"

No olvidar quitar las comillas y reemplazar la ruta, con la ruta donde guardaron el proyecto.

Hay tres archivos que ejecutar para asegurar el correcto funcionamiento del código.

1.- python src/capture_data.py
    Se encarga de obtener las imágenes donde se realizan los gestos, mediante consola se va a pedir que ingrese la palabra o letra para añadir al "diccionario". Hay alrededor de 3-4 segundos disponibles para realizar los gestos para posterior ser almacenados, si no se quiere esperar ese tiempo se puede cancelar o detener de forma temprana la captura de datos presionando la letra "q".

2.- python src/train_model.py
    Se encarga de entrenar al modelo, accediendo al archivo se pueden ajustar los parametros de entrenamiento, como la función de activación, las épocas, entre otros.

3.- python src/gesture_recognition.py
    Es el intérprete, luego de haber ejecutado los dos códigos anteriores, en este se interpretan los gestos que esté realizando el usuario frente a la webcam.


Si se cierra y se desea volver abrir o ejecutar el proyecto, al iniciar la terminal de anaconda aparecerá otro entorno virtual, para activarlo se debe utilizar el mismo código de antes:
    
    conda activate sign_language_env