# Guia de Entrenamiento Profesional de Modelos IA

Sistema completo de entrenamiento de modelos de reconocimiento de animales para Pokedex Ultra.

## Preparacion de Dataset

### Estructura de Directorios Requerida

```
data/training/
├── Perro/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Gato/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Leon/
│   ├── img001.jpg
│   └── ...
└── [Otras especies]/
```

### Requisitos de Imagenes

- **Formato**: JPG o PNG
- **Resolucion minima**: 224x224 pixeles
- **Resolucion recomendada**: 600x600 o superior
- **Imagenes por clase**: Minimo 100, recomendado 500+
- **Total clases**: Minimo 10, sin limite superior

### Descarga de Dataset de Ejemplo

```powershell
# Descargar dataset Animals-10 de Kaggle
kaggle datasets download -d alessiocorrado99/animals10

# Extraer
Expand-Archive animals10.zip -DestinationPath data/training/

# O usar ImageNet subset
```

### Preparacion de Imagenes Propias

Script para procesar imagenes:

```python
import cv2
import os
from pathlib import Path

def prepare_dataset(input_dir, output_dir, target_size=(600, 600)):
    """Procesar y estandarizar imagenes."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        output_class_dir = output_path / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, img_file in enumerate(class_dir.glob("*.*")):
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                    
                img_resized = cv2.resize(img, target_size)
                
                output_file = output_class_dir / f"{class_dir.name}_{idx:04d}.jpg"
                cv2.imwrite(str(output_file), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
            except Exception as e:
                print(f"Error procesando {img_file}: {e}")
                
prepare_dataset("data/raw_images", "data/training")
```

## Entrenamiento de Modelos

### EfficientNetB7 (Recomendado para maxima precision)

```powershell
python train_professional_models.py `
    --data_path "data/training" `
    --model_type efficientnet `
    --epochs 50 `
    --batch_size 16 `
    --learning_rate 0.001 `
    --image_size 600 600
```

**Configuracion avanzada**:

```powershell
python train_professional_models.py `
    --data_path "data/training" `
    --model_type efficientnet `
    --epochs 100 `
    --batch_size 8 `
    --learning_rate 0.0005 `
    --image_size 800 800
```

### ResNet152 (Balance precision/velocidad)

```powershell
python train_professional_models.py `
    --data_path "data/training" `
    --model_type resnet `
    --epochs 50 `
    --batch_size 32 `
    --learning_rate 0.001 `
    --image_size 224 224
```

### MobileNetV2 (Optimizado para velocidad)

```powershell
python train_professional_models.py `
    --data_path "data/training" `
    --model_type mobilenet `
    --epochs 40 `
    --batch_size 64 `
    --learning_rate 0.001 `
    --image_size 224 224
```

## Monitoreo de Entrenamiento

### TensorBoard en Tiempo Real

Abrir terminal adicional:

```powershell
# Activar entorno
.\venv_ultra\Scripts\Activate.ps1

# Iniciar TensorBoard
tensorboard --logdir=data/logs_ultra/tensorboard --port=6006
```

Abrir navegador en: http://localhost:6006

Metricas disponibles:
- Loss de entrenamiento y validacion
- Accuracy de entrenamiento y validacion
- Learning rate
- Distribucion de gradientes
- Activaciones de capas
- Imagenes de ejemplo

### Monitoreo de GPU Durante Entrenamiento

```powershell
# Terminal separada
nvidia-smi -l 1
```

### Logs Detallados

```powershell
# Ver logs en tiempo real
Get-Content training.log -Wait -Tail 50
```

## Data Augmentation

El sistema aplica automaticamente las siguientes transformaciones:

### Transformaciones Geometricas

- Random resized crop (80-100% de la imagen)
- Flip horizontal (50% probabilidad)
- Flip vertical (20% probabilidad)
- Rotacion random (-30 a +30 grados)
- Shift/Scale/Rotate combinado
- Transformacion de perspectiva

### Transformaciones de Color

- Random brightness y contrast
- Hue/Saturation/Value ajustes
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Transformaciones de Ruido

- Gaussian blur
- Gaussian noise
- Coarse dropout (cutout)
- Sharpening

## Estrategia de Entrenamiento

### Fase 1: Entrenamiento del Clasificador

Duracion: 50 epocas (configurable)

1. Base model congelado (pesos de ImageNet)
2. Solo capas del clasificador entrenables
3. Learning rate: 0.001
4. Optimizer: Adam
5. Loss: Sparse Categorical Crossentropy

### Fase 2: Fine-tuning

Duracion: 30 epocas (configurable)

1. Descongelar ultimas 50 capas del base model
2. Learning rate reducido: 0.0001
3. Fine-tuning de caracteristicas de alto nivel

### Callbacks Automaticos

- **ModelCheckpoint**: Guarda mejor modelo segun val_accuracy
- **EarlyStopping**: Detiene si no mejora en 10 epocas
- **ReduceLROnPlateau**: Reduce LR si no mejora en 5 epocas
- **TensorBoard**: Logging de todas las metricas

## Evaluacion de Modelos

### Metricas Calculadas

- **Accuracy**: Precision general
- **Top-5 Accuracy**: Prediccion correcta en top 5
- **Loss**: Funcion de perdida
- **Confusion Matrix**: Matriz de confusion por clase
- **Precision/Recall/F1**: Por clase

### Generar Reporte de Evaluacion

```python
import tensorflow as tf
from train_professional_models import EfficientNetTrainer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar modelo
model = tf.keras.models.load_model('model/ultra/efficientnet_b7_final_20250930_123456.h5')

# Cargar datos de prueba
# ... (codigo de carga de test_ds)

# Predicciones
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# Obtener labels reales
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Reporte de clasificacion
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusion')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
```

## Optimizacion de Hiperparametros

### Grid Search Manual

```powershell
# Probar diferentes learning rates
foreach ($lr in 0.0001, 0.0005, 0.001, 0.005) {
    python train_professional_models.py `
        --data_path "data/training" `
        --model_type efficientnet `
        --learning_rate $lr `
        --epochs 20
}
```

### Batch Size Optimization

Reglas generales:
- **GPU 6GB**: batch_size = 8-16 (EfficientNetB7)
- **GPU 12GB**: batch_size = 16-32
- **GPU 24GB**: batch_size = 32-64

```python
# Test de batch size optimo
import tensorflow as tf

def find_optimal_batch_size(model, start_size=8):
    batch_size = start_size
    while True:
        try:
            # Crear dataset dummy
            dummy_ds = tf.data.Dataset.from_tensor_slices(
                (np.random.rand(batch_size, 600, 600, 3), np.random.randint(0, 10, batch_size))
            ).batch(batch_size)
            
            # Intentar forward pass
            model.fit(dummy_ds, epochs=1, verbose=0)
            
            print(f"Batch size {batch_size}: OK")
            batch_size *= 2
            
        except tf.errors.ResourceExhaustedError:
            print(f"Batch size optimo: {batch_size // 2}")
            break
```

## Exportacion de Modelos

### TensorFlow Lite

Conversion automatica durante entrenamiento.

Manual:

```python
import tensorflow as tf

# Cargar modelo
model = tf.keras.models.load_model('model/ultra/efficientnet_b7_final.h5')

# Convertir
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizaciones
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Quantizacion (opcional, reduce tamano y aumenta velocidad)
converter.target_spec.supported_types = [tf.float16]

# Convertir
tflite_model = converter.convert()

# Guardar
with open('model/ultra/efficientnet_b7.tflite', 'wb') as f:
    f.write(tflite_model)
```

### ONNX (para interoperabilidad)

```python
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model('model/ultra/efficientnet_b7_final.h5')

# Convertir
onnx_model, _ = tf2onnx.convert.from_keras(model)

# Guardar
with open('model/ultra/efficientnet_b7.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

### PyTorch TorchScript

```python
import torch

# Cargar modelo
model = ResNet152Model(num_classes=50)
model.load_state_dict(torch.load('model/ultra/resnet152_best.pth'))
model.eval()

# Convertir
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)

# Guardar
traced_model.save('model/ultra/resnet152.pt')
```

## Transfer Learning Avanzado

### Usar Modelo Pre-entrenado en Dataset Propio

```python
from train_professional_models import EfficientNetTrainer

# Cargar modelo base entrenado
base_model = tf.keras.models.load_model('model/ultra/efficientnet_b7_animals.h5')

# Remover capa de clasificacion
base_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=base_model.layers[-4].output  # Antes de la capa Dense final
)

# Congelar
base_model.trainable = False

# Agregar nuevo clasificador
inputs = tf.keras.Input(shape=(600, 600, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_new_classes, activation='softmax')(x)

new_model = tf.keras.Model(inputs, outputs)

# Entrenar solo el nuevo clasificador
```

## Troubleshooting

### Out of Memory (OOM)

Soluciones:
1. Reducir batch_size
2. Reducir image_size
3. Usar modelo mas pequeno (MobileNet en lugar de EfficientNet)
4. Habilitar memory growth:

```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Overfitting

Sintomas:
- Train accuracy alta, val accuracy baja
- Train loss bajo, val loss alto

Soluciones:
1. Aumentar dropout
2. Mas data augmentation
3. Reducir complejidad del modelo
4. Early stopping mas agresivo
5. Mas datos de entrenamiento

### Underfitting

Sintomas:
- Train y val accuracy bajas
- Loss no disminuye

Soluciones:
1. Modelo mas complejo
2. Mas epocas de entrenamiento
3. Learning rate mas alto
4. Menos regularizacion

### Entrenamiento Muy Lento

Optimizaciones:
1. Verificar uso de GPU: `nvidia-smi`
2. Aumentar batch_size si memoria lo permite
3. Usar mixed precision training:

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

4. Prefetch de datos:

```python
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
```

## Resultados Esperados

### EfficientNetB7

Con dataset de 50 clases, 500 imagenes/clase:

- **Training time**: 8-12 horas (RTX 3060)
- **Validation accuracy**: 92-96%
- **Top-5 accuracy**: 98-99%
- **Inference time**: 80-120ms por imagen

### ResNet152

Con dataset de 50 clases, 500 imagenes/clase:

- **Training time**: 4-6 horas (RTX 3060)
- **Validation accuracy**: 88-93%
- **Top-5 accuracy**: 96-98%
- **Inference time**: 40-60ms por imagen

### MobileNetV2

Con dataset de 50 clases, 500 imagenes/clase:

- **Training time**: 2-3 horas (RTX 3060)
- **Validation accuracy**: 85-90%
- **Top-5 accuracy**: 94-97%
- **Inference time**: 20-30ms por imagen

## Mejores Practicas

1. **Balancear dataset**: Igual numero de imagenes por clase
2. **Validacion cruzada**: k-fold para datasets pequenos
3. **Ensemble de modelos**: Combinar predicciones de multiples modelos
4. **Guardar checkpoints**: Cada N epocas
5. **Monitoreo continuo**: TensorBoard en tiempo real
6. **Documentar experimentos**: Registrar hiperparametros y resultados
7. **Version control**: Git para codigo y DVC para datos/modelos
8. **Testing riguroso**: Conjunto de prueba separado
9. **Calibracion de confianza**: Ajustar umbrales de decision

---

El entrenamiento profesional de modelos requiere experimentacion iterativa. Comenzar con configuracion base y ajustar segun resultados.
