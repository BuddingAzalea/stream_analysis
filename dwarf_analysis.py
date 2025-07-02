import numpy as np
import scipy as sp
import numpy.ma as ma
import numpy.fft as fft
from scipy.interpolate import splrep, BSpline, interp1d
from scipy.stats import binned_statistic
#########################################################################



#########################################################################
"""
DISTANCE CALCULATIONS
"""

def dist(progs):
    """
    takes array of progenetors and computes distances from host 
    for all progenetors
    """
    ar = progs.copy()
    if not (isinstance(ar, np.ndarray)):
        ar = np.array(ar)
    if (len(ar.shape)==2):
        ar = np.array([ar])
    distance = np.linalg.norm(ar[:,:,0:3],axis=-1)   
    return distance

#########################################################################
"""
PERI/APO MASK
"""

def get_peri_mask(progs, distances = None):
    if distances is None:
        distances = dist(progs)
    mask = (distances[:,:-2]>distances[:,1:-1]) & (distances[:,1:-1]<distances[:,2:])
    false_array = np.full(distances.shape[0], False)
    mask = np.column_stack((false_array,mask,false_array))
    return mask, distances

def get_apo_mask(progs, distances = None):
    if distances is None:
        distances = dist(progs)
    mask = (distances[:,:-2]<distances[:,1:-1]) & (distances[:,1:-1]>distances[:,2:])
    false_array = np.full(distances.shape[0], False)
    mask = np.column_stack((false_array,mask,false_array))
    return mask, distances

def get_apo_peri_distances(progs, peri_mask=None, apo_mask=None, distances = None):
    if distances is None:
        distances = dist(progs)
    if peri_mask is None:
        peri_mask, _ = get_peri_mask(progs,distances=distances)
    if apo_mask is None:
        apo_mask, _ = get_apo_mask(progs,distances=distances)
    ex_mask = np.where(peri_mask, -1*distances, 0)
    ex_mask = np.where(apo_mask, distances, ex_mask)
    return ex_mask[0]

def get_data_at_ext(data, peri=True, apo=True, peri_mask=None, apo_mask=None, progs=None, distances=None):
    output = np.empty(data.shape)
    if distances is None:
        distances = dist(progs)
    if peri:
        if peri_mask is None:
            peri_mask, _ = get_peri_mask(progs,distances=distances)
        output += data*peri_mask*-1
    if apo:
        if apo_mask is None:
            apo_mask, _ = get_apo_mask(progs,distances=distances)
        output += data*apo_mask
    return output

def count_peris(progs=None, distances=None, peri_mask=None):
    if peri_mask is None:
        peri_mask = get_peri_mask(progs,distances=distances)
    return np.sum(peri_mask, axis = -1)
    
def count_apos(progs=None, distances=None, apo_mask=None):
    if apo_mask is None:
        apo_mask = get_apo_mask(progs,distances=distances)
    return np.sum(apo_mask, axis = -1)\

#########################################################################












