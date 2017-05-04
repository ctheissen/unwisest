import sys, time
import numpy as np


#####################################################################################

def twoD_Gaussian2((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):

  # Function for a 2-D Gaussian
  # Returns a flattened 1-D array for the Gaussian
  xo = float(xo)
  yo = float(yo)    
  a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
  b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
  c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
  g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                          + c*((y-yo)**2)))
  return g.ravel()

#####################################################################################

def twoD_GaussianU2((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):

  # Function for a 2-D Gaussian
  # Returns an n x n array for the Gaussian
  xo = float(xo)
  yo = float(yo)    
  a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
  b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
  c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
  g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                          + c*((y-yo)**2)))
  return g

#####################################################################################

def eigsorted(cov):

  # Return the eigenvectors, sorted
  vals, vecs = np.linalg.eigh(cov)
  order      = vals.argsort()[::-1]
  return vals[order], vecs[:,order]

#####################################################################################

def FindStd(a,b,x,y,theta,xp,yp):

  # Return the error ellipse in terms of sigma
  A = a**2 * np.sin(theta*np.pi/180)**2 + b**2 * np.cos(theta*np.pi/180)**2
  B = 2 * (b**2 - a**2) * np.sin(theta*np.pi/180) * np.cos(theta*np.pi/180)
  C = a**2 * np.cos(theta*np.pi/180)**2 + b**2 * np.sin(theta*np.pi/180)**2
  D = -2 * A * x - B * y
  E = -B * x - 2 * C * y
  F = A * x**2 + B * x * y + C * y**2 - a**2 * b**2
  std = np.sqrt( (A*xp**2 + B*xp*yp + C*yp **2 + D*xp + E*yp + F + a**2*b**2) / (a**2 * b**2) )
  return std

#####################################################################################




