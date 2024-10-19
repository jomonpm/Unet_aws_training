import tensorflow as tf 
import numpy as np 
from PIL import Image 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.layers import Layer


class ResizeLayer(Layer):
    def __init__(self, target_size):
        super(ResizeLayer, self).__init__()  # Initialize the parent class
        self.target_size = target_size       # Store the target size for resizing

    def call(self, inputs):
        # The inputs argument is 'encoder_val' passed at the time of calling
        return tf.image.resize(inputs, self.target_size)

def convol_relu(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3,3), padding = 'same')(input)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, (3,3), padding = 'same')(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def maxpool(input):
    x = tf.keras.layers.MaxPool2D((2,2))(input)
    return x
    
def up_conv(input, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides = 2, padding = "same")(input)
    #x = convol_relu(x, num_filters)
    return x

def concatination(decoder_val, encoder_val):
    target_size = (decoder_val.shape[1], decoder_val.shape[2])
    #encoder_val = tf.image.resize(encoder_val, size = (decoder_val.shape[1],decoder_val.shape[2])) 
    encoder_val = ResizeLayer(target_size)(encoder_val)

    concat_img = tf.keras.layers.Concatenate()([decoder_val, encoder_val])
    return concat_img


def unet(input_shape):
    input = tf.keras.layers.Input(input_shape) 
    enc1 = convol_relu(input, 64)
    enc1 = maxpool(enc1)

    enc2 = convol_relu(enc1, 128)
    enc2 = maxpool(enc2)

    enc3 = convol_relu(enc2, 256)
    enc3 = maxpool(enc3)

    enc4 = convol_relu(enc3, 512)
    enc4 = maxpool(enc4)

    bridge_in = convol_relu(enc4, 1024)
    bridge_out = up_conv(bridge_in, 1024)
    bridge_out = concatination(bridge_out, enc4)

    dec1 = convol_relu(bridge_out, 512)
    dec1 = up_conv(dec1, 512)
    dec1 = concatination(dec1, enc3)

    dec2 = convol_relu(dec1, 256)
    dec2 = up_conv(dec2, 256)
    dec2 = concatination(dec2, enc2)

    dec3 = convol_relu(dec2, 128)
    dec3 = up_conv(dec3, 128)
    dec3 = concatination(dec3, enc1)

    output = convol_relu(dec3, 64)
    output = tf.keras.layers.Conv2D(1, 1)(output)

    model = tf.keras.models.Model(inputs = input, outputs = output, name = 'U-Net') 
    return model

if __name__ == '__main__': 
    model = unet(input_shape=(512, 512, 3)) 
    model.summary()
'''
img = Image.open('cat.png') 
# Preprocess the image 
img = img.resize((572, 572)) 
img_array = image.img_to_array(img) 
img_array = np.expand_dims(img_array[:,:,:3], axis=0) 
img_array = img_array / 255
  
# Load the model 
model = unet(input_shape=(572, 572, 3)) 
  
# Make predictions 
predictions = model.predict(img_array) 
  
# Convert predictions to a numpy array and resize to original image size 
predictions = np.squeeze(predictions, axis=0) 
predictions = np.argmax(predictions, axis=-1) 
predictions = Image.fromarray(np.uint8(predictions*255)) 
predictions = predictions.resize((img.width, img.height)) 
  
# Save the predicted image 
predictions.save('predicted_image.jpg') 
predictions

'''






