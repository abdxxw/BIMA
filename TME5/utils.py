import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2,fftshift
from PIL import Image
# for interactive ploting, see surf() below

from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D


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
    return fftshift(np.log(1+np.abs(If)))

	
	
	


def sinusoid2d(A, theta, size, T0, Te):
    """ double**2*int*double**2 -> Array[double] """
    ct = np.cos(theta/180*np.pi)
    st = np.sin(theta/180*np.pi)
    x, y = np.meshgrid(np.arange(0, size, Te), np.arange(0, size, Te))
    return A*np.cos(2*np.pi*(y*ct - x*st)/T0)

def shannonInterpolation(I, Te, size):
    """ Array*int*double -> Array[double] """
    n, m = I.shape
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, n))
    Y = np.sinc(x/Te-y)
    x, y = np.meshgrid(np.arange(0, size), np.arange(0, m))
    X = np.sinc(x/Te-y)
    return np.matmul(X.T, np.matmul(I, Y))

def imshow(I,title=None):
    """ display an image """
    plt.figure(figsize=(500//80,500//80))
    plt.gray()
    plt.imshow(I)
    if title: plt.title(title)
    plt.show()

def surf(Z,title=None):
    """ 3D plot of an image """
    X,Y = np.meshgrid(range(Z.shape[1]), range(Z.shape[0]))
    fig = plt.figure(figsize=(600/80,600/80))
    if title: plt.title(title)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
	
	
	
	

def subSample2(I):
    return I[::2,::2]
    
	
def error(Xr,Xd,A,L):
    return np.abs(Xr - Xd).sum() / (2*A* L*L)