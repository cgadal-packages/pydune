# @Author: gadal
# @Date:   2021-02-11T15:11:46+01:00
# @Email:  gadal@ipgp.fr
# @Last modified by:   gadal
# @Last modified time: 2021-02-16T18:38:58+01:00

import numpy as np

############################################################################
################### Trigo functions in degree ##############################
############################################################################

def tand(x):
    return np.tan(x*np.pi/180.)

def sind(x):
    return np.sin(x*np.pi/180.)

def cosd(x):
    return np.cos(x*np.pi/180.)

def atand(x):
    return 180*np.arctan(x)/np.pi

def atan2d(x,y):
    return np.arctan2(x,y)*180/np.pi


############################################################################
########################### Other ##########################################
############################################################################

def Vector_average(Direction, Norm, axis = -1):
    average = np.nanmean(Norm*np.exp(1j*Direction*np.pi/180), axis = axis)
    return np.angle(average)*180/np.pi, np.absolute(average)


def Cartesian_to_polar(X, Y):
    R = np.sqrt(X**2 + Y**2)
    Theta = (np.arctan2(Y, X)*180/np.pi) % 360
    return R, Theta
