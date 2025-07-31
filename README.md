# Reconocimiento OCR de escritura a mano en Python

El reconocimiento óptico de caracteres (OCR) de escritura a mano es un reto interesante y útil en el procesamiento de documentos. Aquí te mostramos un flujo robusto, en cuatro pasos, usando diferentes librerías líderes de Python en cada uno. Aprenderás desde la mejora de la imagen hasta la aplicación de modelos especializados para textos complejos y el uso de soluciones ligeras para casos sencillos o impresos.

## Paso 1: Preprocesamiento de imágenes con OpenCV

python name = step1_preprocesamiento.py

Este paso es fundamental para mejorar la calidad de la imagen de entrada y facilitar la posterior segmentación y reconocimiento del texto manuscrito. 

Se utiliza OpenCV para convertir la imagen a escala de grises, aplicar un suavizado para reducir el ruido, y luego una binarización adaptativa para mejorar el contraste entre el texto y el fondo. 

Finalmente, opcionalmente se aplica una operación morfológica para resaltar los trazos.
```
import cv2                                                      # Librería OpenCV para procesamiento de imágenes
import numpy as np                                              # Numpy para manipulación de matrices

image = cv2.imread('handwriting_sample.jpg')                    # Cargar imagen original
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                  # Convertir a escala de grises
blurred = cv2.GaussianBlur(gray, (5, 5), 0)                     # Suavizar para reducir ruido
thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 21, 8)                               # Binarización adaptativa
kernel = np.ones((2,2), np.uint8)                               # Crear un kernel morfológico pequeño
processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)   # Realzar trazos manuscritos
cv2.imwrite('step1_preprocessed.png', processed)                # Guardar imagen preprocesada
```
## Paso 2: Reconocimiento de escritura a mano mediante EasyOCR

```
python name = step2_easyocr.py

En este paso utilizamos EasyOCR, una librería fácil de usar y con soporte multilenguaje, ideal para obtener buenos resultados en textos manuscritos generales.

Se carga el modelo, se procesa la imagen preprocesada y se imprimen los resultados detectados.

import easyocr                                              # Importar EasyOCR para reconocimiento OCR

reader = easyocr.Reader(['es'], gpu=False)                  # Inicializar lector para español, sin GPU
results = reader.readtext('step1_preprocessed.png')         # Leer texto de la imagen preprocesada

for bbox, text, conf in results:                            # Iterar sobre cada texto detectado
    print(f'Texto: "{text}", Confianza: {conf:.2f}')        # Mostrar texto reconocido y confianza
```

> *Alternativas: es posible usar [keras-ocr](https://github.com/faustomorales/keras-ocr) o [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) según las necesidades de precisión y recursos. Keras-OCR es muy configurable, y TrOCR (de Microsoft) ofrece modelos SOTA de deep learning.*

## Paso 3: Aplicaciones especializadas con Kraken

python name = step3_kraken.py

Kraken es una herramienta avanzada ideal para documentos históricos, multilingües o con escrituras complejas.
Aquí se muestra cómo usar Kraken para segmentar y reconocer líneas de texto, lo que puede mejorar la exactitud en textos difíciles.
Requiere instalar Kraken y modelos entrenados específicos para manuscritos.

```
from kraken import binarization, pageseg, rpred            # Importar módulos clave de Kraken
from PIL import Image                                      # Importar PIL para manejo de imágenes

img = Image.open('handwriting_sample.jpg')                 # Cargar imagen original
bin_img = binarization.nlbin(img)                          # Binarizar imagen con Kraken
segmentation = pageseg.segment(bin_img)                    # Segmentar líneas de texto
model_path = 
	'modelo_manuscrito.mlmodel'                              # Ruta a modelo entrenado (descargar de https://kraken.re/ o entrenar uno propio)
predicter = 
	rpred.PytorchRecognizer.load_model(model_path)           # Cargar modelo OCR

for line in segmentation['lines']:                          # Iterar sobre cada línea segmentada
    crops = bin_img.crop(line.bounds)                       # Recortar línea
    pred = next(rpred.rpred(predicter, crops))              # Reconocer texto en la línea
    print(pred.text)                                        # Imprimir texto reconocido
```

> *Surya es otra alternativa reciente especializada en documentos históricos y scripts complejos.*

## Paso 4: OCR ligero para manuscrita simple o texto impreso (Tesseract + OpenCV)

python name = step4_tesseract.py

Tesseract es el motor OCR más popular para textos impresos y funciona bien con manuscrita sencilla si se preprocesa la imagen adecuadamente.

Aquí se combina OpenCV para limpiar la imagen y Tesseract para extraer el texto.
```
import cv2                                                 # OpenCV para procesar la imagen
import pytesseract                                         # Interfaz Python de Tesseract

img = cv2.imread('handwriting_sample.jpg')                 # Leer imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)               # Convertir a escala de grises

_, thresh = cv2.threshold(gray, 127, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)               # Umbralización para binarizar

config = '--psm 6 -l spa'                                  # Configuración: modo de segmentación y español
text = pytesseract.image_to_string(thresh, config=config)  # Extraer texto con Tesseract
print(text) 									                             # Mostrar el texto extraído
```

## Referencias oficiales


- [OpenCV docs](https://docs.opencv.org/)

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

- [Keras-OCR](https://github.com/faustomorales/keras-ocr)

- [TrOCR (HuggingFace)](https://huggingface.co/docs/transformers/model_doc/trocr)

- [Kraken](https://kraken.re/)

- [Surya](https://github.com/UbiquitousKnowledgeProcessing/Surya)

- [Tesseract OCR](https://tesseract-ocr.github.io/)
