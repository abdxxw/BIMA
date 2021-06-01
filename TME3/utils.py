import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2,fftshift
from PIL import Image



def openImage(fname):
    """ str -> Array 
    (notation above means the function gets a string argument and returns an Array object)
    """
    return np.array(Image.open(fname))
	
	
	
def countPixels(I,k):
    """ Array*int -> int"""

    return np.sum(I == k)

	
def replacePixels(I,k1,k2):
    """ Array*int*int -> Array """
    z = np.copy(I)
    np.where(z==k1, k2, z) 

	
	
def normalizeImage(I,k1,k2):
    """ Array*int*int -> Array """
    maxi = np.max(I)
    mini = np.min(I)
    z = np.copy(I)
    z -= mini
    z = z / (maxi-mini)
    z *= (k2-k1)
    return z + k1
	
def inverteImage(I):
    """ Array -> Array """
    z = np.full(shape=I.size,fill_value=255,dtype=np.int)
    z = z.reshape(I.shape)
    return z-I

	
	
def computeHistogram(I):
    """ Array -> list[int] """
    z = I.flatten()
    out = np.zeros(256)
    for x in z:
        out[int(x)]+=1
    return out
	
def thresholdImage(I,s):
    """ Array*int -> Array """
    z = np.full(shape=I.size,fill_value=255,dtype=np.int)
    i=0
    z1 = I.flatten()
    for x in z1:
        if x < s:
            z[i] = 0
        else:
            z[i] = 255
        i+=1
    return z.reshape(I.shape)
	
def histogramEqualization(I,h):
    """ Array * (list[int] -> Array """
    hc = np.add.accumulate(h)
    I1 = hc[I.flatten().astype(int)]
    I1 = I1 * (np.max(I) / (I.shape[0] * I.shape[1]))
    I1 = I1.astype(int)
    return I1.reshape(I.shape)
	
def computeFT(I):
    """ Array -> Array[complex] """
    return fft2(I)

def toVisualizeFT(If):
    """ Array[complex] -> Array[float] """
    return fftshift(np.abs(If))

def toVisualizeLogFT(If):
    """ Array[complex] -> Array[float] """
    return np.log(1+np.abs(If))

	