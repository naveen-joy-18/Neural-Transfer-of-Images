import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image #type:ignore
from PIL import Image

# Configure TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

def load_and_process_image(img_path, max_dim=512):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (max_dim, max_dim))
    return img

def deprocess_image(img):
    img = img.numpy().squeeze()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def load_style_transfer_model():
    style_transfer_model_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    model = hub.load(style_transfer_model_url)
    return model

def perform_style_transfer(content_image, style_image, model):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

def main(content_image_path, style_image_path, output_image_path):
    # Load and preprocess images
    content_image = load_and_process_image(content_image_path)
    style_image = load_and_process_image(style_image_path)
    
    # Load style transfer model
    model = load_style_transfer_model()
    
    # Apply style transfer
    stylized_image = perform_style_transfer(content_image, style_image, model)
    
    # Save and display the result
    output_image = deprocess_image(stylized_image)
    output_image.save(output_image_path)
    output_image.show()

if __name__ == "__main__":
    # Paths to the images
    content_image_path = 'content.jpg'
    style_image_path = 'style.jpg'
    output_image_path = 'stylized_image.jpg'
     
    main(content_image_path, style_image_path, output_image_path)
