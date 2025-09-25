#!/usr/bin/env python3
"""
Descarga un modelo TFLite y sus labels a la carpeta model/.
Uso:
  python scripts/download_tflite_model.py --model-url <URL_TFLITE> --labels-url <URL_LABELS>
Opciones:
  --model-out  Ruta de salida del modelo (por defecto: model/animal_classifier.tflite)
  --labels-out Ruta de salida de las labels (por defecto: model/labels.txt)
"""

import argparse
import os
import sys
import urllib.request


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Descargando {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-url", required=True)
    p.add_argument("--labels-url", required=True)
    p.add_argument("--model-out", default="model/animal_classifier.tflite")
    p.add_argument("--labels-out", default="model/labels.txt")
    args = p.parse_args()

    try:
        download(args.model_url, args.model_out)
        download(args.labels_url, args.labels_out)
        print("Descargas completadas")
    except Exception as e:
        print(f"Error descargando archivos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
