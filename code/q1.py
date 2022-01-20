# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np

import os
import skimage.io
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """

    image = None
    
    
    w = res[0]
    h = res[1]
    image = np.zeros((w,h))
    x_c = center[0]
    y_c = center[1]
    z_c = center[2]
    r = rad
    
    piw = pxSize * w
    pih = pxSize*h
    piw = piw/2
    pih = pih/2
    
    for i in range(w):
        for j in range(h):
            i_ = i*pxSize -piw
            j_ = j*pxSize-pih
            if(r**2 - (i_-x_c) **2 -(j_-y_c) ** 2>=0):
                
                z = np.sqrt(r**2 - (i_-x_c) **2 -(j_-y_c) ** 2)+z_c
                if (z>0):
                    x = (i_-x_c)
                    y = (j_-y_c)
                
                    n = [2*x , 2*y , 2*z]
                    n = np.array(n)
                    n_hat = n / np.linalg.norm(n)
                    model = np.dot(n_hat,light)
                    image[i][j] = model
                
    
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    
    #im1 = skimage.img_as_float(skimage.io.imread(os.path.join('images',img)))
    images = []
    I = []
    for i in range(1,8):
       
        im1 = imread(path + 'input_' + str(i) + '.tif')
        s = im1.shape[:2]

        im_xyz = rgb2xyz(im1)
        luminance = im_xyz[:,:,1]
        lu = luminance.flatten()


        I.append(lu)
    
    I = np.array(I)
    
    sources = np.load('../data/sources.npy')
    sources = np.array(sources)
    #print (sources.shape)
    L = sources.T
    
    
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    S = L.T
    inv = np.linalg.inv(np.dot(S.T,S))
    B = np.dot(np.dot(inv,S.T),I)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None
    
    albedos = np.linalg.norm(B,axis = 0)
    normals = B/albedos[None,:]
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    
    
    
    albedoIm = albedos.reshape(s)
    #plt.Figure()
    plt.imshow(albedoIm, cmap = 'gray')
    plt.show()
    
    normals = normals.T
    mini = np.abs(np.min(normals))
    # print(mini)
    
    maxm = np.max(normals+mini)
    # print(maxm)
                  
                  
                  
    normals_modi = (normals+mini)/maxm

    normalIm = normals_modi.reshape((s[0],s[1],3))
    print(np.min(normals+mini))
    #normalIm = normals.reshape((s[0],s[1],3))
    #plt.Figure()
    plt.imshow(normalIm, cmap = 'rainbow')
    plt.show()
    
    
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    
    n1 = normals[0,:]
    n2 = normals[1,:]
    n3 = -normals[2,:]
    zx = -n1/n3
    zy = -n2/n3
    zx =zx.reshape(s)
    zy = zy.reshape(s)
    
    surface = integrateFrankot(zx,zy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    surfa = surface.T
    x, y = surfa.shape
    X = np.arange(0, x, 1)
    Y = np.arange(0, y, 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, surface, cmap='coolwarm')
    ax.set_title('Surface plot')
    plt.show()
    pass


if __name__ == '__main__':

    # Put your main code here
    
      #q1.b
      # haha = np.sqrt(3)
      # light1 = np.asarray([1/haha, 1/haha, 1/haha])
      # light2 = np.asarray([1/haha, -1/haha, 1/haha])
      # light3 = np.asarray([-1/haha, -1/haha, 1/haha])
      # c = np.zeros(3)
      # re = np.asarray([3840, 2160])
      # image1 = renderNDotLSphere(center=c, rad=7500, light=light1 , pxSize=7,res=re)
      # plt.imshow(image1.T,origin='lower',cmap='gray')
      # plt.show()
      
      # image2 = renderNDotLSphere(center=c, rad=7500, light=light2,pxSize=7, res=re)
      # plt.imshow(image2.T,origin='lower',cmap='gray')
      # plt.show()
      
      # image3 = renderNDotLSphere(center=c, rad=7500, light=light3,pxSize=7, res=re)
      # plt.imshow(image3.T,origin='lower',cmap='gray')
      # plt.show()
    
    
    
    # u, sv, vh = np.linalg.svd(I, full_matrices=False)
    # print(sv)
    
    I, L, s = loadData(path = "../data/")
    B = estimatePseudonormalsCalibrated(I,L)
    albedos, normals = estimateAlbedosNormals(B)
    sur = estimateShape(normals, s)
    plotSurface(sur)
    #albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
   
    