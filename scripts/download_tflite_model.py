#!/usr/bin/env python3
"""
Script para descargar y configurar modelos TensorFlow Lite optimizados.
Este script descarga modelos preentrenados optimizados para dispositivos edge como Raspberry Pi.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import json
import sys


# URLs de modelos disponibles
MODELS_CONFIG = {
    "mobilenet_v2_1.0_224": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v2_1.0_224_frozen.pb",
        "tflite_url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224.tflite",
        "labels_url": "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v2_1.0_224_info.txt",
        "description": "MobileNet V2 1.0 224x224 - Equilibrio entre precisión y velocidad",
        "input_size": [224, 224, 3],
        "size_mb": 3.4
    },
    "mobilenet_v2_0.75_224": {
        "tflite_url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_0.75_224.tflite",
        "description": "MobileNet V2 0.75 224x224 - Más rápido, menor precisión",
        "input_size": [224, 224, 3],
        "size_mb": 2.6
    },
    "mobilenet_v2_1.4_224": {
        "tflite_url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.4_224.tflite",
        "description": "MobileNet V2 1.4 224x224 - Mayor precisión, más lento",
        "input_size": [224, 224, 3],
        "size_mb": 6.9
    }
}

# Etiquetas ImageNet (incluye muchas clases de animales)
IMAGENET_LABELS = [
    "background",
    "tench, Tinca tinca",
    "goldfish, Carassius auratus", 
    "great white shark, carcharodon carcharias",
    "tiger shark, Galeocerdo cuvier",
    "hammerhead, hammerhead shark",
    "electric ray, crampfish, numbfish, torpedo",
    "stingray",
    "cock",
    "hen",
    "ostrich, Struthio camelus",
    "brambling, Fringilla montifringilla",
    "goldfinch, Carduelis carduelis",
    "house finch, linnet, Carpodacus mexicanus",
    "junco, snowbird",
    "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "robin, American robin, Turdus migratorius",
    "bulbul",
    "jay, blue jay, Cyanocitta cristata",
    "magpie",
    "chickadee",
    "water ouzel, dipper",
    "kite",
    "bald eagle, American eagle, Haliaeetus leucocephalus",
    "vulture",
    "great grey owl, great gray owl, Strix nebulosa",
    "European fire salamander, Salamandra salamandra",
    "common newt, Triturus vulgaris",
    "eft",
    "spotted salamander, Ambystoma maculatum",
    "axolotl, mud puppy, Ambystoma mexicanum",
    "bullfrog, Rana catesbeiana",
    "tree frog, tree-frog",
    "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "loggerhead, loggerhead turtle, Caretta caretta",
    "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
    "mud turtle",
    "terrapin",
    "box turtle, box tortoise",
    "banded gecko",
    "common iguana, iguana, Iguana iguana",
    "American chameleon, anole, Anolis carolinensis",
    "whiptail, whiptail lizard",
    "agama",
    "frilled lizard, Chlamydosaurus kingi",
    "alligator lizard",
    "Gila monster, Heloderma suspectum",
    "green lizard, Lacerta viridis",
    "African chameleon, Chamaeleo chamaeleon",
    "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "African crocodile, Nile crocodile, Crocodylus niloticus",
    "American alligator, Alligator mississipiensis",
    "triceratops",
    "thunder snake, worm snake, Carphophis amoenus",
    "ringneck snake, ring-necked snake, ring snake",
    "hognose snake, puff adder, sand viper",
    "green snake, grass snake",
    "king snake, kingsnake",
    "garter snake, grass snake",
    "water snake",
    "vine snake",
    "night snake, Hypsiglena torquata",
    "boa constrictor, Constrictor constrictor",
    "rock python, rock snake, Python sebae",
    "Indian cobra, Naja naja",
    "green mamba",
    "sea snake",
    "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "diamondback, diamondback rattlesnake, Crotalus adamanteus",
    "sidewinder, horned rattlesnake, Crotalus cerastes",
    "trilobite",
    "harvestman, daddy longlegs, Phalangium opilio",
    "scorpion",
    "black and gold garden spider, Argiope aurantia",
    "barn spider, Araneus cavaticus",
    "garden spider, Aranea diademata",
    "black widow, Latrodectus mactans",
    "tarantula",
    "wolf spider, hunting spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse, partridge, Bonasa umbellus",
    "prairie chicken, prairie grouse, prairie fowl",
    "peacock",
    "quail",
    "partridge",
    "African grey, African gray, Psittacus erithacus",
    "macaw",
    "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser, Mergus serrator",
    "goose",
    "black swan, Cygnus atratus",
    "tusker",
    "echidna, spiny anteater, anteater",
    "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
    "wallaby, brush kangaroo",
    "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    "wombat",
    "jellyfish",
    "sea anemone, anemone",
    "brain coral",
    "flatworm, platyhelminth",
    "nematode, nematode worm, roundworm",
    "conch",
    "snail",
    "slug",
    "sea slug, nudibranch",
    "chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "chambered nautilus, pearly nautilus, nautilus",
    "Dungeness crab, Cancer magister",
    "rock crab, Cancer irroratus",
    "fiddler crab",
    "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
    "American lobster, Northern lobster, Maine lobster, Homarus americanus",
    "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "crayfish, crawfish, crawdad, crawdaddy",
    "hermit crab",
    "isopod",
    "white stork, Ciconia ciconia",
    "black stork, Ciconia nigra",
    "spoonbill",
    "flamingo",
    "little blue heron, Egretta caerulea",
    "American egret, great white heron, Egretta albus",
    "bittern",
    "crane",
    "limpkin, Aramus pictus",
    "European gallinule, Porphyrio porphyrio",
    "American coot, marsh hen, mud hen, water hen, Fulica americana",
    "bustard",
    "ruddy turnstone, Arenaria interpres",
    "red-backed sandpiper, dunlin, Erolia alpina",
    "redshank, Tringa totanus",
    "dowitcher",
    "oystercatcher, oyster catcher",
    "pelican",
    "king penguin, Aptenodytes patagonica",
    "albatross, mollymawk",
    "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
    "killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
    "dugong, Dugong dugon",
    "sea lion",
    "Chihuahua",
    "Japanese spaniel",
    "Maltese dog, Maltese terrier, Maltese",
    "Pekinese, Pekingese, Peke",
    "Shih-Tzu",
    "Blenheim spaniel",
    "papillon",
    "toy terrier",
    "Rhodesian ridgeback",
    "Afghan hound, Afghan",
    "basset, basset hound",
    "beagle",
    "bloodhound, sleuthhound",
    "bluetick",
    "black-and-tan coonhound",
    "Walker hound, Walker foxhound",
    "English foxhound",
    "redbone",
    "borzoi, Russian wolfhound",
    "Irish wolfhound",
    "Italian greyhound",
    "whippet",
    "Ibizan hound, Ibizan Podenco",
    "Norwegian elkhound, elkhound",
    "otterhound, otter hound",
    "Saluki, gazelle hound",
    "Scottish deerhound, deerhound",
    "Weimaraner",
    "Staffordshire bullterrier, Staffordshire bull terrier",
    "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
    "Bedlington terrier",
    "Border terrier",
    "Kerry blue terrier",
    "Irish terrier",
    "Norfolk terrier",
    "Norwich terrier",
    "Yorkshire terrier",
    "wire-haired fox terrier",
    "Lakeland terrier",
    "Sealyham terrier, Sealyham",
    "Airedale, Airedale terrier",
    "cairn, cairn terrier",
    "Australian terrier",
    "Dandie Dinmont, Dandie Dinmont terrier",
    "Boston bull, Boston terrier",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "Scotch terrier, Scottish terrier, Scottie",
    "Tibetan terrier, chrysanthemum dog",
    "silky terrier, Sydney silky",
    "soft-coated wheaten terrier",
    "West Highland white terrier",
    "Lhasa, Lhasa apso",
    "flat-coated retriever",
    "curly-coated retriever",
    "golden retriever",
    "Labrador retriever",
    "Chesapeake Bay retriever",
    "German short-haired pointer",
    "vizsla, Hungarian pointer",
    "English setter",
    "Irish setter, red setter",
    "Gordon setter",
    "Brittany spaniel",
    "clumber, clumber spaniel",
    "English springer, English springer spaniel",
    "Welsh springer spaniel",
    "cocker spaniel, English cocker spaniel, cocker",
    "Sussex spaniel",
    "Irish water spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old English sheepdog, bobtail",
    "Shetland sheepdog, Shetland sheep dog, Shetland",
    "collie",
    "Border collie",
    "Bouvier des Flandres, Bouviers des Flandres",
    "Rottweiler",
    "German shepherd, German shepherd dog, German police dog, alsatian",
    "Doberman, Doberman pinscher",
    "miniature pinscher",
    "Greater Swiss Mountain dog",
    "Bernese mountain dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull mastiff",
    "Tibetan mastiff",
    "French bulldog",
    "Great Dane",
    "Saint Bernard, St Bernard",
    "Eskimo dog, husky",
    "malamute, malemute, Alaskan malamute",
    "Siberian husky",
    "dalmatian, coach dog, carriage dog",
    "affenpinscher, monkey pinscher, monkey dog",
    "basenji",
    "pug, pug-dog",
    "Leonberg",
    "Newfoundland, Newfoundland dog",
    "Great Pyrenees",
    "Samoyed, Samoyede",
    "Pomeranian",
    "chow, chow chow",
    "keeshond",
    "Brabancon griffon",
    "Pembroke, Pembroke Welsh corgi",
    "Cardigan, Cardigan Welsh corgi",
    "toy poodle",
    "miniature poodle",
    "standard poodle",
    "Mexican hairless",
    "timber wolf, grey wolf, gray wolf, Canis lupus",
    "white wolf, Arctic wolf, Canis lupus tundrarum",
    "red wolf, maned wolf, Canis rufus, Canis niger",
    "coyote, prairie wolf, brush wolf, Canis latrans",
    "dingo, warrigal, warragal, Canis dingo",
    "dhole, Cuon alpinus",
    "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
    "hyena, hyaena",
    "red fox, Vulpes vulpes",
    "kit fox, Vulpes macrotis",
    "Arctic fox, white fox, Alopex lagopus",
    "grey fox, gray fox, Urocyon cinereoargenteus",
    "tabby, tabby cat",
    "tiger cat",
    "Persian cat",
    "Siamese cat, Siamese",
    "Egyptian cat",
    "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    "lynx, catamount",
    "leopard, Panthera pardus",
    "snow leopard, ounce, Panthera uncia",
    "jaguar, panther, Panthera onca, Felis onca",
    "lion, king of beasts, Panthera leo",
    "tiger, Panthera tigris",
    "cheetah, chetah, Acinonyx jubatus",
    "brown bear, bruin, Ursus arctos",
    "American black bear, black bear, Ursus americanus, Euarctos americanus",
    "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
    "sloth bear, Melursus ursinus, Ursus ursinus",
    "mongoose",
    "meerkat, mierkat",
    "tiger beetle",
    "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "ground beetle, carabid beetle",
    "long-horned beetle, longicorn, longicorn beetle",
    "leaf beetle, chrysomelid",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant, emmet, pismire",
    "grasshopper, hopper",
    "cricket",
    "walking stick, walkingstick, stick insect",
    "cockroach, roach",
    "mantis, mantid",
    "cicada, cicala",
    "leafhopper",
    "lacewing, lacewing fly",
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    "damselfly",
    "admiral",
    "ringlet, ringlet butterfly",
    "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    "cabbage butterfly",
    "sulphur butterfly, sulfur butterfly",
    "lycaenid, lycaenid butterfly",
    "starfish, sea star",
    "sea urchin",
    "sea cucumber, holothurian",
    "wood rabbit, cottontail, cottontail rabbit",
    "hare",
    "Angora, Angora rabbit",
    "hamster",
    "porcupine, hedgehog",
    "fox squirrel, eastern fox squirrel, Sciurus niger",
    "marmot",
    "beaver",
    "guinea pig, Cavia cobaya",
    "sorrel",
    "zebra",
    "hog, pig, grunter, squealer, Sus scrofa",
    "wild boar, boar, Sus scrofa",
    "warthog",
    "hippopotamus, hippo, river horse, Hippopotamus amphibius",
    "ox",
    "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
    "bison",
    "ram, tup",
    "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    "ibex, Capra ibex",
    "hartebeest",
    "impala, Aepyceros melampus",
    "gazelle",
    "Arabian camel, dromedary, Camelus dromedarius",
    "llama",
    "weasel",
    "mink",
    "polecat, fitch, foulmart, foumart, Mustela putorius",
    "black-footed ferret, ferret, Mustela nigripes",
    "otter",
    "skunk, polecat, wood pussy",
    "badger",
    "armadillo",
    "three-toed sloth, ai, Bradypus tridactylus",
    "orangutan, orang, orangutang, Pongo pygmaeus",
    "gorilla, Gorilla gorilla",
    "chimpanzee, chimp, Pan troglodytes",
    "gibbon, Hylobates lar",
    "siamang, Hylobates syndactylus, Symphalangus syndactylus",
    "guenon, guenon monkey",
    "patas, hussar monkey, Erythrocebus patas",
    "baboon",
    "macaque",
    "langur",
    "colobus, colobus monkey",
    "proboscis monkey, Nasalis larvatus",
    "marmoset",
    "capuchin, ringtail, Cebus capucinus",
    "howler monkey, howler",
    "titi, titi monkey",
    "spider monkey, Ateles geoffroyi",
    "squirrel monkey, Saimiri sciureus",
    "Madagascar cat, ring-tailed lemur, Lemur catta",
    "indri, indris, Indri indri, Indri brevicaudatus",
    "Indian elephant, Elephas maximus",
    "African elephant, Loxodonta africana",
    "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
    "barracouta, snoek",
    "eel",
    "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
    "rock beauty, Holocanthus tricolor",
    "anemone fish",
    "sturgeon",
    "gar, garfish, garpike, billfish, Lepisosteus osseus",
    "lionfish",
    "puffer, pufferfish, blowfish, globefish"
]


def create_model_dir():
    """Crear directorio de modelos si no existe."""
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    return model_dir


def download_file(url, destination):
    """Descargar archivo con barra de progreso."""
    print(f"Descargando desde: {url}")
    print(f"Guardando en: {destination}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgreso: {percent}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)", end="")
        else:
            print(f"\rDescargado: {downloaded/1024/1024:.1f}MB", end="")
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print()  # Nueva línea después de la barra de progreso
        return True
    except Exception as e:
        print(f"\nError descargando: {e}")
        return False


def create_labels_file(model_dir):
    """Crear archivo de etiquetas ImageNet."""
    labels_path = model_dir / "labels.txt"
    
    print("Creando archivo de etiquetas...")
    
    # Filtrar solo las primeras 400 etiquetas que incluyen la mayoría de animales
    animal_labels = IMAGENET_LABELS[:400]
    
    with open(labels_path, "w", encoding="utf-8") as f:
        for label in animal_labels:
            f.write(f"{label}\n")
    
    print(f"Archivo de etiquetas creado: {labels_path}")
    return labels_path


def convert_keras_to_tflite():
    """Convertir modelo Keras existente a TensorFlow Lite."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        keras_model_path = "model/animal_classifier.h5"
        tflite_model_path = "model/animal_classifier.tflite"
        
        if not os.path.exists(keras_model_path):
            print(f"Modelo Keras no encontrado en {keras_model_path}")
            return False
        
        print("Convirtiendo modelo Keras a TensorFlow Lite...")
        
        # Cargar modelo Keras
        model = keras.models.load_model(keras_model_path)
        
        # Convertir a TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimización por defecto
        
        # Configuraciones adicionales para mejor compatibilidad con Pi
        converter.representative_dataset = None
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        
        # Guardar modelo TensorFlow Lite
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"Modelo TensorFlow Lite guardado en: {tflite_model_path}")
        
        # Mostrar información del modelo convertido
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Información del modelo TensorFlow Lite:")
        print(f"- Input shape: {input_details[0]['shape']}")
        print(f"- Input dtype: {input_details[0]['dtype']}")
        print(f"- Output shape: {output_details[0]['shape']}")
        print(f"- Tamaño del archivo: {os.path.getsize(tflite_model_path) / 1024 / 1024:.1f} MB")
        
        return True
        
    except ImportError:
        print("TensorFlow no disponible para conversión")
        return False
    except Exception as e:
        print(f"Error en conversión: {e}")
        return False


def download_pretrained_model(model_name):
    """Descargar modelo preentrenado."""
    if model_name not in MODELS_CONFIG:
        print(f"Modelo '{model_name}' no disponible")
        print("Modelos disponibles:")
        for name, config in MODELS_CONFIG.items():
            print(f"- {name}: {config['description']}")
        return False
    
    model_dir = create_model_dir()
    config = MODELS_CONFIG[model_name]
    
    # Descargar modelo TensorFlow Lite
    if "tflite_url" in config:
        tflite_path = model_dir / "animal_classifier.tflite"
        print(f"Descargando {model_name} ({config['size_mb']} MB)...")
        
        if download_file(config["tflite_url"], tflite_path):
            print(f"Modelo descargado exitosamente")
            
            # Crear archivo de etiquetas
            create_labels_file(model_dir)
            
            # Crear archivo de información
            info_path = model_dir / "model_info.json"
            model_info = {
                "model_name": model_name,
                "description": config["description"],
                "input_size": config["input_size"],
                "size_mb": config["size_mb"],
                "download_date": str(Path().cwd()),
                "tflite_path": str(tflite_path.relative_to(Path().cwd())),
                "labels_path": "model/labels.txt"
            }
            
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2)
            
            print(f"Información del modelo guardada en: {info_path}")
            return True
        else:
            return False
    else:
        print(f"URL de TensorFlow Lite no disponible para {model_name}")
        return False


def list_available_models():
    """Listar modelos disponibles."""
    print("Modelos TensorFlow Lite disponibles:")
    print("-" * 50)
    
    for name, config in MODELS_CONFIG.items():
        print(f"Model: {name}")
        print(f"   Descripción: {config['description']}")
        print(f"   Tamaño de entrada: {config['input_size']}")
        print(f"   Tamaño archivo: {config['size_mb']} MB")
        print()


def main():
    """Función principal del script."""
    print("=" * 60)
    print("DESCARGADOR DE MODELOS TENSORFLOW LITE")
    print("Pokédx Animal - Optimizado para Raspberry Pi")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Uso:")
        print(f"  python {sys.argv[0]} list                    # Listar modelos disponibles")
        print(f"  python {sys.argv[0]} download <modelo>       # Descargar modelo específico")
        print(f"  python {sys.argv[0]} convert                 # Convertir modelo Keras existente")
        print()
        list_available_models()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_available_models()
        
    elif command == "download":
        if len(sys.argv) < 3:
            print("Especifica el nombre del modelo a descargar")
            list_available_models()
            return
        
        model_name = sys.argv[2]
        success = download_pretrained_model(model_name)
        
        if success:
            print()
            print("Descarga completada exitosamente!")
            print("El modelo está listo para usar en Raspberry Pi")
        else:
            print("Error en la descarga")
    
    elif command == "convert":
        print("Convirtiendo modelo Keras existente a TensorFlow Lite...")
        success = convert_keras_to_tflite()
        
        if success:
            print("Conversión completada!")
            create_labels_file(create_model_dir())
        else:
            print("Error en la conversión")
    
    else:
        print(f"Comando desconocido: {command}")
        print("Comandos disponibles: list, download, convert")


if __name__ == "__main__":
    main()
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
