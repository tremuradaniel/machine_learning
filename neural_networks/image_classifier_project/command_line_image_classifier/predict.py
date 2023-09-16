import argparse

parser = argparse.ArgumentParser(description='Image Classifier Command Line App')

parser.add_argument('path_to_image', action="store")
parser.add_argument('saved_model', action="store")
parser.add_argument('--top_k', action="store", type=int, default=3)
parser.add_argument('--category_names', action="store")

print('importing libraries...')
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import json
print('done importing libraries!')

args = parser.parse_args()

print("loading model...")
loaded_model = tf.keras.models.load_model(
  (f"./{args.saved_model}"),
   custom_objects={'KerasLayer':hub.KerasLayer}
)
print("model loaded!")
print("processing image...")
def process_image(image):
  image = tf.convert_to_tensor(image)
  image = tf.image.resize(image, (224, 224))
  image = tf.cast(image, tf.float32)
  image /= 255
  return image.numpy()

def predict(image_path, model, top_k):
  image = Image.open(image_path)
  image = np.asarray(image)
  image = process_image(image)
  image = np.expand_dims(image, axis=0)
  prediction = model.predict(image)
  probabilities, classes = tf.math.top_k(prediction, k=top_k)
  
  return probabilities.numpy()[0], classes.numpy()[0]

def printClassNames(classesNo):
  if args.category_names is not None:
    result = []
    for classNo in classesNo:
      classIndex = str(classNo+1)
      result.append(class_names[classIndex])
    print(result)
      
  

if args.category_names is not None:
  with open(args.category_names, 'r') as f:
        class_names = json.load(f)

image_path = args.path_to_image

probs, classes = predict(image_path, loaded_model, args.top_k)
print("done!")
print(probs)
print(classes)
printClassNames(classes)


