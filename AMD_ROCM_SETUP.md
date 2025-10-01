# Guia de Instalacion GPU AMD con ROCm - Windows

Guia completa para configurar Pokedex Animal Ultra con aceleracion GPU AMD en Windows.

## Requisitos del Sistema

### Hardware

- GPU: AMD Radeon RX 6700 XT o superior
- CPU: Compatible con AVX2
- RAM: 16 GB minimo, 32 GB recomendado
- Almacenamiento: 100 GB SSD libre

### Software

- Windows 10/11 (64-bit)
- Python 3.10 o 3.11 (ROCm no soporta 3.12 completamente)
- Visual Studio 2019/2022 Build Tools
- AMD Adrenalin Drivers actualizados

## Instalacion de ROCm en Windows

### Opcion 1: PyTorch con ROCm (RECOMENDADO)

PyTorch tiene mejor soporte ROCm en Windows que TensorFlow.

```powershell
# Desinstalar PyTorch existente
pip uninstall torch torchvision torchaudio

# Instalar PyTorch con ROCm 5.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Verificar instalacion
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Opcion 2: TensorFlow-DirectML (Alternativa)

DirectML funciona con GPUs AMD en Windows.

```powershell
# Desinstalar TensorFlow existente
pip uninstall tensorflow tensorflow-gpu

# Instalar TensorFlow-DirectML
pip install tensorflow-directml

# Verificar
python -c "import tensorflow as tf; print(f'GPUs disponibles: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

### Opcion 3: ROCm Nativo (Avanzado - Linux WSL2)

Para maximo rendimiento, usar WSL2 con Ubuntu.

```bash
# En WSL2 Ubuntu
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo dpkg -i amdgpu-install_5.7.50700-1_all.deb
sudo amdgpu-install -y --usecase=rocm

# Agregar usuario al grupo
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Reiniciar
sudo reboot

# Instalar PyTorch con ROCm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## Configuracion del Proyecto

### Paso 1: Verificar GPU

```powershell
python utils/gpu_detector.py
```

Salida esperada:
```
============================================================
CONFIGURACION DE GPU
============================================================
Tipo: AMD
Nombre: AMD Radeon RX 6700 XT
Dispositivo: cuda
ROCm: Disponible
============================================================
```

### Paso 2: Variables de Entorno

Agregar al PATH del sistema:

```powershell
# PowerShell como Administrador
[Environment]::SetEnvironmentVariable("HSA_OVERRIDE_GFX_VERSION", "10.3.0", "Machine")
[Environment]::SetEnvironmentVariable("PYTORCH_ROCM_ARCH", "gfx1031", "Machine")
```

Para RX 6700 XT:
- GFX Version: 10.3.0
- Architecture: gfx1031

### Paso 3: Instalar Dependencias Adicionales

```powershell
pip install wmi pywin32
```

### Paso 4: Actualizar requirements

Editar `requirements_windows_ultra_py312.txt`:

```
# Para AMD GPU
torch>=2.1.0
torchvision>=0.16.0
# Usar index ROCm:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# TensorFlow con DirectML
tensorflow-directml>=1.15.8

# Resto de dependencias...
```

## Optimizaciones de Rendimiento

### PyTorch

```python
import torch

# Habilitar optimizaciones
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Precision mixta (si soportada)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### TensorFlow-DirectML

```python
import tensorflow as tf

# Configurar memoria
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Configuracion de Cache

```python
# En pokedex_ultra_windows.py
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512'
```

## Verificacion de Rendimiento

### Script de Benchmark

```python
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# Test de rendimiento
tensor_size = (1000, 1000)
iterations = 100

tensor_a = torch.randn(tensor_size).to(device)
tensor_b = torch.randn(tensor_size).to(device)

start = time.time()
for _ in range(iterations):
    result = torch.matmul(tensor_a, tensor_b)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()

print(f"Tiempo: {end - start:.3f}s")
print(f"GFLOPS: {(2 * tensor_size[0]**3 * iterations) / (end - start) / 1e9:.2f}")
```

Rendimiento esperado RX 6700 XT:
- GFLOPS: 200-400 (dependiendo de configuracion)
- Tiempo: 0.5-1.0s

## Resolucion de Problemas

### Error: "No GPU disponible"

```powershell
# Verificar drivers
dxdiag

# Actualizar Adrenalin
# Descargar de: https://www.amd.com/en/support

# Verificar PyTorch
python -c "import torch; print(torch.__version__); print(torch.version.hip if hasattr(torch.version, 'hip') else 'No ROCm')"
```

### Error: "HIP error"

```powershell
# Limpiar cache
python -c "import torch; torch.cuda.empty_cache()"

# Verificar version HIP
hipcc --version
```

### Rendimiento Bajo

```python
# Habilitar profiling
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Tu codigo aqui
    model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memoria GPU Insuficiente

```python
# Reducir batch size en train_professional_models.py
batch_size = 8  # Reducir de 16

# Habilitar gradient checkpointing
model.gradient_checkpointing_enable()

# Limpiar cache periodicamente
import gc
gc.collect()
torch.cuda.empty_cache()
```

## Configuracion Avanzada

### Multi-GPU (si disponible)

```python
import torch.nn as nn

# DataParallel
model = nn.DataParallel(model)

# DistributedDataParallel (mejor rendimiento)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
model = DDP(model)
```

### Precision Mixta

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Monitoreo de GPU

### Script de Monitoreo

```python
import GPUtil
import psutil
import time

while True:
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        print(f"GPU: {gpu.load*100:.1f}% | Mem: {gpu.memoryUsed}/{gpu.memoryTotal}MB | Temp: {gpu.temperature}C")
    
    time.sleep(1)
```

### Herramientas Recomendadas

- AMD Radeon Software: Monitoreo en tiempo real
- GPU-Z: Informacion detallada
- HWiNFO: Sensores y metricas
- TensorBoard: Metricas de entrenamiento

## Benchmarks Esperados

### RX 6700 XT (12GB VRAM)

#### Inferencia

- MobileNetV2: 150-200 FPS
- EfficientNetB7: 30-40 FPS
- YOLOv8x: 40-60 FPS

#### Entrenamiento

- Batch 16, EfficientNetB7: 8-12 iter/s
- Batch 32, MobileNetV2: 25-35 iter/s
- Batch 8, YOLOv8x: 5-8 iter/s

## Referencias

### Documentacion Oficial

- ROCm: https://rocm.docs.amd.com/
- PyTorch ROCm: https://pytorch.org/get-started/locally/
- TensorFlow-DirectML: https://github.com/microsoft/tensorflow-directml

### Comunidad

- ROCm GitHub: https://github.com/RadeonOpenCompute/ROCm
- PyTorch Forums: https://discuss.pytorch.org/
- AMD Developer Community: https://community.amd.com/

### Versiones Compatibles

- RX 6700 XT: ROCm 5.4+
- PyTorch: 2.1.0+ con ROCm 5.7
- TensorFlow-DirectML: 1.15.8+

## Notas Importantes

1. ROCm en Windows es experimental. Para produccion, usar Linux.
2. TensorFlow-DirectML es mas estable en Windows que TensorFlow-ROCm.
3. PyTorch con ROCm tiene mejor rendimiento que DirectML.
4. Actualizar drivers AMD regularmente.
5. Monitorear temperaturas GPU durante entrenamiento.

## Proximos Pasos

Despues de configurar GPU:

```powershell
# Verificar sistema completo
python verify_system_ultra.py

# Ejecutar aplicacion
python pokedex_ultra_windows.py

# Entrenar modelos
python train_professional_models.py --dataset data/training --epochs 50
```

El sistema detectara automaticamente la GPU AMD y usara el backend apropiado.
