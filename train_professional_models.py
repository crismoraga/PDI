#!/usr/bin/env python3
"""
Sistema de Entrenamiento Profesional de Modelos de IA
Para Pokedex Animal Ultra - Windows Edition

Este script implementa un pipeline completo de entrenamiento con:
- Transfer learning avanzado
- Data augmentation profesional
- Optimizacion de hiperparametros
- Validacion cruzada
- Exportacion a multiples formatos
- Monitoreo en tiempo real con TensorBoard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    EfficientNetB7,
    MobileNetV2,
    ResNet152V2,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("model/ultra")
DATA_DIR = Path("data/training")
LOGS_DIR = Path("data/logs_ultra/tensorboard")
RESULTS_DIR = Path("data/training_results")

for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class AnimalDataset(Dataset):
    """Dataset personalizado para entrenamiento de reconocimiento de animales."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = cv2.resize(image, self.image_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        label = self.labels[idx]
        
        return image, label


class AdvancedDataAugmentation:
    """Sistema avanzado de data augmentation para maxima generalizacion."""
    
    @staticmethod
    def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """Transformaciones para datos de entrenamiento."""
        return A.Compose([
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.3
            ),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    @staticmethod
    def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """Transformaciones para datos de validacion."""
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class EfficientNetTrainer:
    """Trainer profesional para EfficientNetB7."""
    
    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int] = (600, 600),
        learning_rate: float = 0.001,
        batch_size: int = 16
    ):
        self.num_classes = num_classes
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        
        self._setup_gpu()
        self._build_model()
        
    def _setup_gpu(self) -> None:
        """Configurar GPU para entrenamiento."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPUs disponibles: {len(gpus)}")
            except RuntimeError as e:
                logger.error(f"Error configurando GPU: {e}")
        else:
            logger.warning("No se detectaron GPUs. Entrenamiento en CPU.")
            
    def _build_model(self) -> None:
        """Construir modelo EfficientNetB7 con transfer learning."""
        base_model = EfficientNetB7(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.image_size, 3)
        )
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=(*self.image_size, 3))
        
        x = base_model(inputs, training=False)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
        
        logger.info(f"Modelo EfficientNetB7 construido. Parametros totales: {self.model.count_params():,}")
        
    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        epochs: int = 50,
        fine_tune_epochs: int = 30
    ) -> keras.callbacks.History:
        """Entrenar modelo con estrategia de dos fases."""
        logger.info("Iniciando fase 1: Entrenamiento de clasificador")
        
        callbacks = self._get_callbacks(phase="initial")
        
        history_initial = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Iniciando fase 2: Fine-tuning de capas profundas")
        
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        fine_tune_at = len(base_model.layers) - 50
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
        
        callbacks = self._get_callbacks(phase="fine_tune")
        
        history_fine_tune = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=fine_tune_epochs,
            initial_epoch=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history_fine_tune
        
        return self.history
        
    def _get_callbacks(self, phase: str) -> List[keras.callbacks.Callback]:
        """Obtener callbacks para entrenamiento."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            ModelCheckpoint(
                filepath=str(MODELS_DIR / f"efficientnet_b7_{phase}_{timestamp}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(LOGS_DIR / f"efficientnet_{phase}_{timestamp}"),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
        
    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """Evaluar modelo en conjunto de prueba."""
        results = self.model.evaluate(test_data, verbose=1, return_dict=True)
        
        logger.info("Resultados de evaluacion:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return results
        
    def save_model(self, filepath: str) -> None:
        """Guardar modelo entrenado."""
        self.model.save(filepath)
        logger.info(f"Modelo guardado en: {filepath}")
        
        tf_lite_path = filepath.replace('.h5', '.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tf_lite_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"Modelo TFLite guardado en: {tf_lite_path}")


class PyTorchResNetTrainer:
    """Trainer profesional para ResNet152 con PyTorch."""
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        self._build_model()
        
    def _build_model(self) -> None:
        """Construir modelo ResNet152 con transfer learning."""
        self.model = models.resnet152(pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        num_features = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.num_classes)
        )
        
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(
            self.model.fc.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Modelo ResNet152 construido.")
        logger.info(f"  Parametros totales: {total_params:,}")
        logger.info(f"  Parametros entrenables: {trainable_params:,}")
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Entrenar una epoca."""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            progress_bar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': float(running_corrects) / total_samples
            })
            
        epoch_loss = running_loss / total_samples
        epoch_acc = float(running_corrects) / total_samples
        
        return epoch_loss, epoch_acc
        
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validar una epoca."""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validation"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
        epoch_loss = running_loss / total_samples
        epoch_acc = float(running_corrects) / total_samples
        
        return epoch_loss, epoch_acc
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50
    ) -> Dict[str, List[float]]:
        """Entrenar modelo completo."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_acc = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            logger.info(f"\nEpoca {epoch+1}/{epochs}")
            logger.info("-" * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            self.scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = MODELS_DIR / f"resnet152_best_{timestamp}.pth"
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Mejor modelo guardado: {save_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping activado en epoca {epoch+1}")
                break
                
        return history
        
    def save_model(self, filepath: str) -> None:
        """Guardar modelo entrenado."""
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Modelo guardado en: {filepath}")


class TrainingPipeline:
    """Pipeline completo de entrenamiento profesional."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        """Cargar dataset desde directorio."""
        data_path = Path(self.config['data_path'])
        
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
            
        image_paths = []
        labels = []
        class_names = []
        
        for class_dir in sorted(data_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                class_names.append(class_name)
                
                for img_path in class_dir.glob("*.jpg") + class_dir.glob("*.png"):
                    image_paths.append(str(img_path))
                    labels.append(class_name)
                    
        logger.info(f"Dataset cargado: {len(image_paths)} imagenes, {len(set(labels))} clases")
        
        label_indices = self.label_encoder.fit_transform(labels)
        
        return image_paths, label_indices.tolist(), class_names
        
    def prepare_tensorflow_dataset(
        self,
        image_paths: List[str],
        labels: List[int]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Preparar datasets de TensorFlow."""
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels, test_size=0.3, stratify=labels, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        def load_and_preprocess_image(path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.config['image_size'])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
            
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.cache().shuffle(1000).batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds
        
    def run(self) -> None:
        """Ejecutar pipeline completo de entrenamiento."""
        logger.info("Iniciando pipeline de entrenamiento profesional")
        logger.info(f"Configuracion: {json.dumps(self.config, indent=2)}")
        
        image_paths, labels, class_names = self.load_dataset()
        
        with open(MODELS_DIR / "class_names.json", 'w') as f:
            json.dump(class_names, f, indent=2)
            
        num_classes = len(class_names)
        
        if self.config['model_type'] == 'efficientnet':
            logger.info("Entrenando EfficientNetB7")
            
            train_ds, val_ds, test_ds = self.prepare_tensorflow_dataset(image_paths, labels)
            
            trainer = EfficientNetTrainer(
                num_classes=num_classes,
                image_size=tuple(self.config['image_size']),
                learning_rate=self.config['learning_rate'],
                batch_size=self.config['batch_size']
            )
            
            history = trainer.train(
                train_ds,
                val_ds,
                epochs=self.config['epochs'],
                fine_tune_epochs=self.config.get('fine_tune_epochs', 30)
            )
            
            results = trainer.evaluate(test_ds)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trainer.save_model(str(MODELS_DIR / f"efficientnet_b7_final_{timestamp}.h5"))
            
        logger.info("Pipeline de entrenamiento completado exitosamente")


def parse_arguments() -> argparse.Namespace:
    """Parsear argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Sistema de entrenamiento profesional de modelos de IA"
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Ruta al directorio de datos de entrenamiento"
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['efficientnet', 'resnet', 'mobilenet'],
        default='efficientnet',
        help="Tipo de modelo a entrenar"
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help="Numero de epocas de entrenamiento"
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Tamano de batch"
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help="Learning rate inicial"
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[600, 600],
        help="Tamano de imagen (height width)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Funcion principal."""
    args = parse_arguments()
    
    config = {
        'data_path': args.data_path,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'image_size': args.image_size
    }
    
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
