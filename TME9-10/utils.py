import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2,fftshift
from PIL import Image
# for interactive ploting, see surf() below

from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d

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
    z = np.full(shape=I.size,fill_value=1,dtype=np.int)
    i=0
    z1 = I.flatten()
    for x in z1:
        if x < s:
            z[i] = 0
        else:
            z[i] = 1
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
	
def niceDisplay14_bis(affichages,titres=None):
    """ list[Array]*list[str] -> NoneType
        display from 1 up to 4 images or vectors with optionnal titles
        2D arrays are displayed as image with imshow()
        1D arrays are displayed as curve with plot()
    """
    
    if not type(affichages) == type([]):
        affichages = [affichages]
        
    if titres is None:
        titres = ['',]*len(affichages)
        
    if not type(titres) == type([]):
        titres = [titres]
    
    nb_affichages = len(affichages)
    if nb_affichages >5 or nb_affichages < 1 :
        raise ValueError('niceDisplay_14 : affichage should be a list of length 1 up to 4')
        
    if nb_affichages != len(titres):
        raise ValueError('niceDisplay_14 : titres must have same length than affichage')
    plt.gray()
    fig , ax = plt.subplots(1,nb_affichages)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    for i in range(0,nb_affichages):
        ax[i].imshow(affichages[i])
        ax[i].set_title(titres[i])

    plt.show()


	
	
	

def subSample2(I):
    return I[::2,::2]
    
	
def error(Xr,Xd,A,L):
    return np.abs(Xr - Xd).sum() / (2*A* L*L)
	
	
def imagePad(I,h):
    
    n, m = I.shape
    pad = int(h.shape[0]/2)
    out = np.full((n+2*pad,m+2*pad), 0.0)
    out[pad:pad+m, pad:pad+n] = I
    return out


	
def orientation(Ix, Iy, Ig):
    """ Array[n,m]**3 -> Array[n,m]
        Returns an image of orientation.
    """
    n, m = Ix.shape
    x = np.arange(4)*np.pi/4
    ori = np.stack((np.cos(x), np.sin(x)), axis=1)
    O = np.zeros(Ix.shape)
    for i in range(n):
        for j in range(m):
            if Ig[i, j] > 0:
                v = np.array([Ix[i, j], -Iy[i, j]])/Ig[i, j]
                if Iy[i, j] > 0: v = -v
                prod = np.matmul(ori, v)
                maxi = prod.max()
                imax = np.nonzero(prod == maxi)
                O[i, j] = imax[0][0]+1
    return O

def gaussianKernel(sigma):
    """ double -> Array
        return a gaussian kernel of standard deviation sigma
    """
    n2 = np.int(3*sigma)
    x,y = np.meshgrid(np.arange(-n2,n2+1),np.arange(-n2,n2+1))
    kern =  np.exp(-(x**2+y**2)/(2*sigma*sigma))
    return kern/kern.sum()

def returnSobelDirv(I):
    Sx = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    Sy = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
    Ix = convolve2d(I,Sx,mode='same')
    Iy = convolve2d(I,Sy,mode='same')
    Ig = np.sqrt(Ix **2+Iy**2)
    return Ix,Iy,Ig
	
def SobelDetector(I, s):
    """ Array*double -> Array """
    Ix, Iy, Ig = returnSobelDirv(I)
    return np.where(Ig <= s,0,1)
	
def LaplacianDetector(I, s):
    """ Array*double -> Array """
    L = np.asarray([[0,1,-0],[1,-4,1],[0,1,0]])
    IL = convolve2d(I,L,mode='same')
    out = np.zeros(IL.shape)

    for x in range(1, out.shape[0] - 1):
        for y in range(1, out.shape[1] - 1):
            neighbors = IL[x-1:x+2, y-1:y+2]
            maxP = neighbors.max()
            minP = neighbors.min()

            if ((maxP - minP) > s) and minP <0 and maxP >0:
                out[x, y] = 1
    return np.asarray(out)
	
	
def nms(Ig, Ior):
    """ Array**2 -> Array """
    out = np.zeros(Ig.shape)
    for i in range(1,Ig.shape[0]-1):
        for j in range(1,Ig.shape[1]-1):
            if Ior[i,j] == 0: 
                out[i,j] = Ig[i,j]
                continue
            if Ior[i,j] == 3: 
                r = Ig[i+1,j]
                q = Ig[i-1,j]
            if Ior[i,j] == 4: 
                r = Ig[i+1,j+1]
                q = Ig[i-1,j-1]
            if Ior[i,j] == 1: 
                r = Ig[i,j-1]
                q = Ig[i,j+1]
            if Ior[i,j] == 2: 
                r = Ig[i+1,j-1]
                q = Ig[i-1,j+1]
            if Ig[i,j] >= r and Ig[i,j] >= q:
                out[i,j] = Ig[i,j]
    return out
            