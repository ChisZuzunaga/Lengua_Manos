# Lengua Manos

**Lengua Manos** es un mega proyecto enfocado en el reconocimiento de gestos mediante el entrenamiento de redes neuronales para interpretar lenguajes de señas. Este proyecto permite capturar datos de gestos, entrenar modelos de reconocimiento y realizar interpretaciones en tiempo real.

## Tecnologías Utilizadas

El proyecto está desarrollado principalmente con:

- **Python** (99.2%): Para el desarrollo de los modelos, procesamiento de datos y lógica del programa.
- **Batchfile** (0.8%): Scripts para automatización de tareas y configuraciones.

## Estructura del Proyecto

La carpeta que almacenará el proyecto debe seguir esta estructura:

```
/sign_language_translator
│
├── /data                 # Almacena las imágenes de los gestos y letras
│   ├── /letra_A          # Carpeta para la letra A
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   ├── /letra_B          # Carpeta para la letra B
│   │   ├── img1.png
│   │   └── ...
│   └── ...               # Otras letras y gestos
│
├── /src                  # Código fuente
│   ├── capture_data.py   # Captura de datos de gestos/letras
│   ├── train_model.py    # Entrenamiento del modelo
│   └── gesture_recognition.py # Reconocimiento de gestos/letras
│
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Descripción del proyecto
```

## Instalación y Configuración

### Requisitos previos

- Tener [Anaconda](https://www.anaconda.com/) instalado.

### Pasos para configurar el entorno

1. Crear un entorno virtual en Anaconda:
   ```bash
   conda create --name sign_language_env python=3.8
   ```

2. Activar el entorno virtual:
   ```bash
   conda activate sign_language_env
   ```

3. Instalar las dependencias necesarias:
   ```bash
   conda install tensorflow
   conda install tensorflow-gpu
   conda install opencv
   pip install mediapipe
   conda install numpy
   ```

4. Cambiar a la ruta del proyecto:
   ```bash
   cd C:\Users\queso\OneDrive\Escritorio\Lengua_Manos
   ```
   > Reemplaza la ruta con la ubicación donde guardaste el proyecto.

## Uso del Proyecto

Hay tres archivos principales que debes ejecutar para asegurar el correcto funcionamiento del código:

1. **Captura de datos**:
   ```bash
   python src/capture_data.py
   ```
   - Permite capturar imágenes de los gestos y almacenarlas.
   - Durante la ejecución, se te pedirá que ingreses la palabra o letra que deseas añadir al "diccionario".
   - Tendrás 3-4 segundos para realizar gestos que serán almacenados como imágenes. Para detener la captura antes de tiempo, presiona `q`.

2. **Entrenamiento del modelo**:
   ```bash
   python src/train_model.py
   ```
   - Entrena el modelo de reconocimiento con los datos capturados.
   - Puedes ajustar parámetros como la función de activación, las épocas, entre otros, directamente en el archivo.

3. **Reconocimiento de gestos**:
   ```bash
   python src/gesture_recognition.py
   ```
   - Interpreta los gestos realizados frente a la webcam, utilizando el modelo entrenado.

### Nota

Si cierras y deseas reabrir el proyecto, recuerda activar el entorno virtual nuevamente:
```bash
conda activate sign_language_env
```

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas colaborar en este proyecto:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tus cambios:
   ```bash
   git checkout -b mi-nueva-funcionalidad
   ```
3. Realiza tus cambios y súbelos:
   ```bash
   git commit -m "Añadida nueva funcionalidad"
   git push origin mi-nueva-funcionalidad
   ```
4. Envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.

---

Si tienes preguntas o sugerencias, no dudes en abrir un issue en el repositorio.
