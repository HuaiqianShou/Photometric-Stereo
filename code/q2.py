# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt
import scipy

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    # B = None
    # L = None
    # u, sv, vh = np.linalg.svd(I, full_matrices=False)
    # sv[3:]=0
    # print(sv)
    # L = np.dot(u,sv)
    
    # vh = vh[:3,:]
    # B = vh
    # return B, L
    B = None
    L = None
    u, sv, vh = np.linalg.svd(I, full_matrices=False)
    sv[3:]=0
    sv = sv[:3]
    # sv = sv[:3].reshape(3,1)
    # sv = np.concatenate((sv, sv, sv),axis = 1)
    sv = np.diag(sv)
    
    sv =  scipy.linalg.fractional_matrix_power(sv,1/2)
    L = np.dot(u[:,:3],sv)

    vh = vh[:3,:]
    B = np.dot(sv,vh)
    return B, L.T


def basrelief(B,u,v,lamda):
    
    G = [[1,0,0],[0,1,0],[u,v,lamda]]
    print(G)
    G = np.array(G)
    G_inv = np.linalg.inv(G)
    GB = np.dot(G_inv.T,B)
    
    
    return GB

if __name__ == "__main__":

    # Put your main code here
    
    I, L0, s = loadData(path = "../data/")
    B,L = estimatePseudonormalsUncalibrated(I)
    B_enforce = enforceIntegrability(B, s, sig = 3)
    
    B_enforce = basrelief(B_enforce,-5,5,0.01)
    albedos, normals = estimateAlbedosNormals(B_enforce)
    # albedos, normals = estimateAlbedosNormals(B)
    # albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    # surface = estimateShape(normals, s)
    # plotSurface(surface)
    
    #albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    surface = estimateShape(normals, s)
    
    plotSurface(surface)

