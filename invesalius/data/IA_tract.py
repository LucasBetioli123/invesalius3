
import tensorflow as tf
import numpy as np
from nibabel.affines import apply_affine
from dipy.io.image import load_nifti
from tensorflow.keras.models import load_model
import random
from scipy import ndimage
t1 = r"E:\thais_7T_MRI\Superficies\dados\T1_thais.nii"




def transformtract_nifit(dataframe_arr_index):
    tract_data = dataframe_arr_index.to_numpy()
    data, affine = load_nifti(t1)
    data = np.array(data)
    for i in range(data.shape[0]):
	    for j in range(data.shape[1]):
		    for k in range(data.shape[2]):
			    data[i, j, k] = 0
    xyz=tract_data[:,0:3]
    xyz = apply_affine(np.linalg.inv(affine), xyz)
    x = xyz[:, 0].reshape(-1, 1).astype(int)
    y = xyz[:, 1].reshape(-1, 1).astype(int)
    z = xyz[:, 2].reshape(-1, 1).astype(int)
    
    red = tract_data[:,3:4]
    green =tract_data[:,4:5]
    blue = tract_data[:,5:6]
    opacity = tract_data[:, 6:7]
    gray = (0.299 * red + 0 * 587 * green + 0.114 * blue) * opacity  # transforma os 3 canais em um canal cinza
    gray = gray.astype(int)
    data[x, y, z] = gray[:]
    
    return data
    
def setfloat(volume):
	"""set volumn as float32"""
	volume = volume.astype("float32")
	return volume
def load_model_trained():
	model = load_model(r"C:\Users\Lucas biomag\Documents\GitHub\invesalius3\invesalius\data\Lenet_gray.h5")
	return model
	

def resize_volume(img):
	"""Resize across z-axis"""
	# Set the desired depth
	desired_depth = 64
	desired_width = 128
	desired_height = 128
	# Get current depth
	current_depth = img.shape[-1]
	current_width = img.shape[0]
	current_height = img.shape[1]
	# Compute depth factor
	depth = current_depth / desired_depth
	width = current_width / desired_width
	height = current_height / desired_height
	depth_factor = 1 / depth
	width_factor = 1 / width
	height_factor = 1 / height
	
	img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
	img = np.expand_dims(img, axis=-1)  # Add channel dimension
	#
	# Add batch dimension
	img = np.expand_dims(img, axis=0)
	return img
def preprocessing(volume):
	"""Read and resize to volume"""

	
	# Resize width, height and depth
	volume = resize_volume(volume)
	return volume

def rotate(volume):
	"""Rotate the volume by a few degrees"""

	def scipy_rotate(volume):
		#define some rotation angles
		angles = [-20, -10, -5, 5, 10, 20]
		# pick angles at random
		angle = random.choice(angles)
		# rotate volume
		volume = ndimage.rotate(volume, angle, reshape=False)

		return volume

	augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
	return augmented_volume
def processing(volume):
	"""Process training data by rotating and adding a channel."""
	# Rotate volume
	volume = rotate(volume)
	#volume = tf.expand_dims(volume, axis=3)
	return volume

def predicition(volume_processed):
	model=load_model_trained()
	prediction=model.predict(volume_processed)
	return prediction
    
    
    
    