import sys, time
from astropy.table import Table, join, vstack
from astropy.io import ascii, fits
import numpy as np
import scipy as sp
from scipy import signal
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.stats as astats
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from astropy.utils.data import download_file
from astropy.utils.data import clear_download_cache
from astropy.visualization import ZScaleInterval
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, ICRS
import utilities as util

############################
#### Core functionality ####
############################

def unwisest(ra, dec, plot=True, verbose=True, outfile=None, pixel=True, neo=True, show_progress=True, savefig=False):

  # Set some default values
  d2a        = 3600.
  d2ma       = 360000.
  a2d        = 1 / d2a
  platescale = 0.000763888888889 # degrees to WISE pixels
  thres      = 3

  if plot:
    # Set some plotting functions
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('text', usetex=True)
    plt.rc('legend', fontsize=7)
    plt.rc('axes', labelsize=10)
    figcount   = 1  

  Ra0, Dec0 = 22.83646, -6.86806

  # Download the unWISE tile list
  File0 = download_file('http://unwise.me/data/tiles.fits', cache=False, show_progress=show_progress)
  T1    = Table.read(File0)

  # Find the closest tile that is cenetered most on the object
  c1      = SkyCoord(Ra0*u.degree, Dec0*u.degree, frame='icrs')
  c2      = SkyCoord(T1['ra']*u.degree, T1['dec']*u.degree, frame='icrs')
  sep     = c1.separation(c2)
  distRA  = abs((Ra0 - T1['ra'].data) * np.cos(Dec0*np.pi/180))
  distDEC = abs(Dec0 - T1['dec'].data)
  ind     = np.where( (sep.degree < 1.15) & (distRA < .8) & (distDEC < .8) )[0]
  coaddid = T1['coadd_id'][ind].data[0]

  # Initialize arrays to keep results for each band
  Positions  = []
  PositionsP = []
  Amplitudes = []
  Eccent     = []

  # Loop through bands
  for wband in ['w1','w2','w3','w4']:

    if verbose:
      print('\nBand: %s'%wband.upper())

    # Do neoWISE if requested
    if neo == True and wband in ['w1', 'w2']:

      url = 'http://unwise.me/data/neo1/%s/%s/unwise-%s-%s-img-m.fits'%(coaddid[0:3], coaddid, coaddid, wband)
      filename = download_file(url, cache=False, show_progress=show_progress)
      File     = fits.open(filename)
      t        = File[0].data
      w1       = WCS(filename)

      url = 'http://unwise.me/data/neo1/%s/%s/unwise-%s-%s-invvar-m.fits.gz'%(coaddid[0:3], coaddid, coaddid, wband)
      filename = download_file(url, cache=False, show_progress=show_progress)
      File     = fits.open(filename)
      t2       = 1./np.sqrt(File[0].data)

    else: # Do regular unWISE

      url = 'http://unwise.me/data/%s/%s/unwise-%s-%s-img-m.fits'%(coaddid[0:3], coaddid, coaddid, wband)
      filename = download_file(url, cache=False, show_progress=show_progress)
      File     = fits.open(filename)
      t        = File[0].data
      w1       = WCS(filename)

      url = 'http://unwise.me/data/%s/%s/unwise-%s-%s-invvar-m.fits.gz'%(coaddid[0:3], coaddid, coaddid, wband)
      filename = download_file(url, cache=False, show_progress=show_progress)
      File     = fits.open(filename)
      t2       = 1./np.sqrt(File[0].data)

    # Create empty arrays for fitting
    xA, yA   = np.mgrid[0:t.shape[0]:1, 0:t.shape[1]:1]
    xAf, yAf = np.mgrid[0:t.shape[0]:100j, 0:t.shape[1]:100j]

    # Convert pixel positions into RADEC positions
    xA2, yA2 = w1.wcs_pix2world(xA, yA, 0)
    xA3, yA3 = w1.wcs_pix2world(xA-0.5, yA-0.5, 0)

    # Grab a box around where we care
    boxsize = 60 # Size of the box in arcsec (default is 60 arcsec)
    Ra1     = Ra0  - boxsize*a2d / np.cos(Dec0 * np.pi/180.)
    Ra2     = Ra0  + boxsize*a2d / np.cos(Dec0 * np.pi/180.)
    Dec1    = Dec0 - boxsize*a2d
    Dec2    = Dec0 + boxsize*a2d

    # Convert boxsize from RADEC (world coordinates) to pixels
    Px0, Py0 = w1.wcs_world2pix(Ra0, Dec0, 0)
    Px1, Py1 = w1.wcs_world2pix(Ra1, Dec1, 0)
    Px2, Py2 = w1.wcs_world2pix(Ra2, Dec2, 0)

    # Make an array of min/max pixel coordinates
    Xs = np.array([Px1, Px2])
    Ys = np.array([Py1, Py2])

    # Create a meshgrid of pixel coordinates from the world coordinates
    x, y   = np.mgrid[int(np.floor(np.min(Ys))):int(np.ceil(np.max(Ys))):1, int(np.floor(np.min(Xs))):int(np.ceil(np.max(Xs))):1]
    xf, yf = np.mgrid[int(np.floor(np.min(Ys))):int(np.ceil(np.max(Ys))):100j, int(np.floor(np.min(Xs))):int(np.ceil(np.max(Xs))):100j]

    # Convert pixel coordinates to world coordinates
    x2, y2 = w1.wcs_pix2world(y, x, 0)
    x3, y3 = w1.wcs_pix2world(y-0.5, x-0.5, 0)

    # Grab the data for the box
    ydata  = t[int(np.floor(np.min(Ys))):int(np.ceil(np.max(Ys))), int(np.floor(np.min(Xs))):int(np.ceil(np.max(Xs)))]
    ydataU = t2[int(np.floor(np.min(Ys))):int(np.ceil(np.max(Ys))), int(np.floor(np.min(Xs))):int(np.ceil(np.max(Xs)))]

    # Start the fitting
    p0         = [10, Py0, Px0, 1, 1, 0]       # Set the initial guess
    bounds     = [ [0, Py0-5, Px0-5, 0, 0, 0], # Set the lower and upper limits for each parameter
                   [10000000, Py0+5, Px0+5, 20, 20, 2*np.pi] ] 
    # Fit the parameters
    popt, pcov = curve_fit(util.twoD_Gaussian2, (x, y), ydata.ravel(), p0=p0, sigma=ydataU.ravel(), bounds=bounds, max_nfev=1000000)

    # Convert pixel coordinates to world coordinates
    RAT1, DECT1 = w1.wcs_pix2world(popt[2], popt[1], 0)

    # Print positional statements if wished
    if verbose:
      print('Input Position (R.A., Dec.):', Ra0, Dec0)
      print('Extracted Position (R.A., Dec.):', RAT1, DECT1)
      print('Offsets (R.A., Dec.) [arcsec]:', abs(RAT1 - Ra0)*d2a * np.cos(Dec0 * np.pi/180.), abs(DECT1 - Dec0)*d2a)

    #  store positions for end of the run comparison between bands
    Positions.append( (RAT1, DECT1) )
    PositionsP.append( (popt[2], popt[1]) )
    Amplitudes.append(popt[0])

    # Try to do the sigmas in arcsec (more accuate than using the platescale, which is not a constant)
    RAT0, DECT0 = w1.wcs_pix2world(popt[2], popt[1], 0)
    RAT2, DECT  = w1.wcs_pix2world(popt[2]+popt[4], popt[1], 0)
    RAT, DECT2  = w1.wcs_pix2world(popt[2], popt[1]+popt[3], 0)

    # Put the sigmas into an array to compute eccentricities
    Ax          = np.array([popt[3],  popt[4]])
    Eccent.append(np.sqrt(1. - (np.min(Ax)**2/ np.max(Ax)**2) ))

    sigR0 = abs(RAT0 - RAT2)*d2a  * np.cos(DECT0 * np.pi/180.)
    sigD0 = abs(DECT0 - DECT2)*d2a

    # Print some pertinent details if wished
    if verbose:
      print('std_x, std_y (pixel):', popt[4],  popt[3])
      print('std_R.A., std_Dec. (arcsec):', sigR0, sigD0)
      #print 'Sig (FWHM):', sigR0 * np.sqrt(8 * np.log(2)), sigD0 * np.sqrt(8 * np.log(2))#, np.sqrt(8 * np.log(2)) 
      print('Eccentricity (pixel):', np.sqrt(1. - (np.min(Ax)**2/ np.max(Ax)**2) ))
      print('Roundness (pixel) [Cotten et al. 2016]:', (popt[3] - popt[4]) / ( (popt[3] + popt[4]) / 2.))

    # Write to a file
    if outfile != None:
      target = open(outfile, 'a')
      target.write('%s, %s, %s, %s, %s, %s, %s, %s, %s\n'%(Ra0, Dec0, wband, popt[0], popt[2], popt[1], popt[4], popt[3], popt[5]) )
      target.close

    # Plot it if wished
    if plot:

      # Initialize the figure/subfigures
      left, width    = 0.1, 0.65
      bottom, height = 0.1, 0.65
      bottom_h       = left_h = left + width + 0.02

      rect_main      = [left,   bottom,   width, height]
      rect_x         = [left,   bottom_h, width, 0.15]
      rect_y         = [left_h, bottom,   0.15,  height]


      ########## ZSCALE
      interval = ZScaleInterval()
      vmin, vmax = interval.get_limits(ydata)
      #print 'ZSCALE:', vmin, vmax
      ########## ZSCALE

      if pixel: # Plot in pixel space

        plt.figure(figcount, figsize=(6,6))
        #plt.figure(figcount, figsize=(3.4,3.4))
        #plt.title('%s'%wband)
        axm = plt.axes(rect_main, aspect='equal')
        axx = plt.axes(rect_x, sharex=axm)
        axy = plt.axes(rect_y, sharey=axm)

        axm.minorticks_on()
        axx.minorticks_on()
        axy.minorticks_on()

        axm.annotate(r'\textbf{%s}'%str(wband).upper(),
              xy=(.82, .84), xycoords='figure fraction',
              horizontalalignment='center', verticalalignment='center',
              fontsize=12)
     
        axm.pcolormesh(xA-0.5, yA-0.5, t.T, cmap='Greys', vmin=vmin, vmax=vmax, rasterized=True, alpha=0.8)
       
        try: 
          axm.contour(yf, xf, util.twoD_GaussianU2((xf,yf), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), 1, colors='r', alpha=0.5, rasterized=True)
          axm.contour(yf, xf, util.twoD_GaussianU2((xf,yf), popt[0], popt[1], popt[2], 2*popt[3], 2*popt[4], popt[5]), 1, colors='r', alpha=0.5, rasterized=True)
        except: 
          print 'ERROR - Cannot plot gaussian contours.'

        axm.set_xlim(Px2, Px1)
        axm.set_ylim(Py1, Py2)
        xmin, xmax = axm.get_xlim()

        axx.plot(np.arange(int(np.around(np.min(Xs))), int(np.around(np.max(Xs))), 1), t.T[int(np.around(np.min(Xs))):int(np.around(np.max(Xs))), int(np.around(popt[1]))])
        axx.plot(np.arange(int(np.around(np.min(Xs))), int(np.around(np.max(Xs))), .01), util.twoD_GaussianU2((int(np.around(popt[1])), np.arange(int(np.around(np.min(Xs))), int(np.around(np.max(Xs))), .01)), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), alpha=0.5, c='r', ls='-')

        axy.plot(t.T[int(np.around(popt[2])), int(np.around(np.min(Ys))):int(np.around(np.max(Ys)))], np.arange(int(np.around(np.min(Ys))), int(np.around(np.max(Ys))), 1))
        axy.plot(util.twoD_GaussianU2( ( np.arange(int(np.around(np.min(Ys))), int(np.around(np.max(Ys))), .01), int(np.around(popt[2])) ), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), np.arange(int(np.around(np.min(Ys))), int(np.around(np.max(Ys))), .01), alpha=0.5, c='r', ls='-')

        axm.axvline(popt[2], c='r', ls='--', alpha=0.5)
        axm.axhline(popt[1], c='r', ls='--', alpha=0.5)
        axx.axvline(popt[2], c='r', ls='--', alpha=0.5)
        axy.axhline(popt[1], c='r', ls='--', alpha=0.5)

        axm.set_xlabel('R.A. (pixel)')
        axm.set_ylabel('Decl. (pixel)')


      else: # Plot in RADEC space

        plt.figure(figcount, figsize=(6,6))

        axm = plt.axes(rect_main)
        axx = plt.axes(rect_x, sharex=axm)
        axy = plt.axes(rect_y, sharey=axm)

        at0 = AnchoredText(r'\textbf{%s}'%str(wband).upper(),
                    prop=dict(size=15, va='bottom', color='k'), frameon=False,
                    loc=1,
                    )
        axm.add_artist(at0)

        axm.pcolormesh(xA3, yA3, t.T, cmap='Greys_r', vmin=vmin, vmax = vmax)
        #xT, yT = w1.wcs_pix2world(y, x, 0)
        try:
          axm.contour(x2, y2, util.twoD_GaussianU2((x,y), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), 1, colors='r', alpha=0.5)
          axm.contour(x2, y2, util.twoD_GaussianU2((x,y), popt[0], popt[1], popt[2], 2*popt[3], 2*popt[4], popt[5]), 1, colors='r', alpha=0.5)
        except: 
          print 'ERROR - Cannot plot gaussian contours.'

        #axm.set_xlim(Px2, Px1)
        #axm.set_ylim(Py1, Py2)
        axm.set_xlim(Ra2, Ra1)
        axm.set_ylim(Dec1, Dec2)
        xmin, xmax = axm.get_xlim()

        axx.plot(xA2[int(np.around(np.min(Xs))):int(np.around(np.max(Xs))), int(np.around(popt[1]))], t.T[int(np.around(np.min(Xs))):int(np.around(np.max(Xs))), int(np.around(Py0))])
        #axx.plot(np.arange(xmin, xmax, 1), t.T[xmin:xmax, int(np.around(Py0))], drawstyle='steps-mid')
        #Xplot1 = np.arange(int(np.around(np.min(Xs))), int(np.around(np.max(Xs))), .01))
        Xplot1 = np.linspace(int(np.around(np.min(xA[int(np.around(np.min(Xs))):int(np.around(np.max(Xs))), int(np.around(Py0))]))), int(np.around(np.max(xA[int(np.around(np.min(Xs))):int(np.around(np.max(Xs))), int(np.around(Py0))]))), 1000)
        Xplot2 = w1.wcs_pix2world(Xplot1, int(np.around(popt[1])), 0)[0]
        axx.plot(Xplot2, util.twoD_GaussianU2((int(np.around(popt[1])), Xplot1), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), alpha=0.7, c='r', ls=':')

        axy.plot(t.T[int(np.around(popt[2])), int(np.around(np.min(Ys))):int(np.around(np.max(Ys)))], yA2[int(np.around(Px0)), int(np.around(np.min(Ys))):int(np.around(np.max(Ys)))])    
        #Yplot1 = np.arange(int(np.around(np.min(Ys))), int(np.around(np.max(Ys))), .01)
        Yplot1 = np.linspace(int(np.around(np.min(yA[int(np.around(Px0)), int(np.around(np.min(Ys))):int(np.around(np.max(Ys)))]))), int(np.around(np.max(yA[int(np.around(Px0)), int(np.around(np.min(Ys))):int(np.around(np.max(Ys)))]))), 1000)
        Yplot2 = w1.wcs_pix2world(int(np.around(Px0)), Yplot1, 0)[1]
        axy.plot(util.twoD_GaussianU2( ( Yplot1, int(np.around(popt[2])) ), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]), Yplot2, alpha=0.7, c='r', ls=':')

        axm.axvline(RAT0, c='r', ls='--', alpha=0.5)
        axm.axhline(DECT0, c='r', ls='--', alpha=0.5)
        axx.axvline(RAT0, c='r', ls='--', alpha=0.5)
        axy.axhline(DECT0, c='r', ls='--', alpha=0.5)

        axm.set_xlabel('R.A. (deg.)')
        axm.set_ylabel('Decl. (deg.)')

    axm.annotate('', (.95, .85), (.95, 1), xycoords='axes fraction', arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
    axm.annotate('N', (.89, .94), xycoords='axes fraction', size=8)
    axm.annotate('', (.95, .85), (.8, .85), xycoords='axes fraction', arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
    axm.annotate('E', (.82, .88), xycoords='axes fraction', size=8)

    axx.xaxis.tick_top()
    axy.yaxis.tick_right()
    figcount += 1

    if savefig:
      #plt.savefig('%s.png'%wband, dpi=600, bbox_inches='tight')
      plt.savefig('%s.pdf'%wband, dpi=600, bbox_inches='tight')
      plt.savefig('%0.4f_%0.4f_%s.pdf'%(Ra0, Dec0, wband), bbox_inches='tight')
   
  if verbose:
    # print 'OFFSETS (arcsec):'
    # print 'Bands\t R.A.\t Dec.'
    # print 'W1-W2\t %0.3f\t %0.3f'%( (Positions[0][0] - Positions[1][0]) * d2a * np.cos(Positions[0][1] * np.pi/180.), (Positions[0][1] - Positions[1][1]) * d2a)
    # print 'W1-W3\t %0.3f\t %0.3f'%( (Positions[0][0] - Positions[2][0]) * d2a * np.cos(Positions[0][1] * np.pi/180.), (Positions[0][1] - Positions[2][1]) * d2a)
    # print 'W1-W4\t %0.3f\t %0.3f'%( (Positions[0][0] - Positions[3][0]) * d2a * np.cos(Positions[0][1] * np.pi/180.), (Positions[0][1] - Positions[3][1]) * d2a)
    # print 'W2-W3\t %0.3f\t %0.3f'%( (Positions[1][0] - Positions[2][0]) * d2a * np.cos(Positions[1][1] * np.pi/180.), (Positions[1][1] - Positions[2][1]) * d2a)
    # print 'W2-W4\t %0.3f\t %0.3f'%( (Positions[1][0] - Positions[3][0]) * d2a * np.cos(Positions[1][1] * np.pi/180.), (Positions[1][1] - Positions[3][1]) * d2a)
    # print 'W3-W4\t %0.3f\t %0.3f'%( (Positions[2][0] - Positions[3][0]) * d2a * np.cos(Positions[2][1] * np.pi/180.), (Positions[2][1] - Positions[3][1]) * d2a)

    # print 'OFFSETS (pixel):'
    # print 'Bands\t R.A.\t Dec.'
    # print 'W1-W2\t %0.3f\t %0.3f'%( (PositionsP[0][0] - PositionsP[1][0]), (PositionsP[0][1] - PositionsP[1][1]))
    # print 'W1-W3\t %0.3f\t %0.3f'%( (PositionsP[0][0] - PositionsP[2][0]), (PositionsP[0][1] - PositionsP[2][1]))
    # print 'W1-W4\t %0.3f\t %0.3f'%( (PositionsP[0][0] - PositionsP[3][0]), (PositionsP[0][1] - PositionsP[3][1]))
    # print 'W2-W3\t %0.3f\t %0.3f'%( (PositionsP[1][0] - PositionsP[2][0]), (PositionsP[1][1] - PositionsP[2][1]))
    # print 'W2-W4\t %0.3f\t %0.3f'%( (PositionsP[1][0] - PositionsP[3][0]), (PositionsP[1][1] - PositionsP[3][1]))
    # print 'W3-W4\t %0.3f\t %0.3f'%( (PositionsP[2][0] - PositionsP[3][0]), (PositionsP[2][1] - PositionsP[3][1]))
    # print ''

    print('\nOffsets:')
    print('Bands\tR.A. (pix)\tDec. (pix)\tR.A. (asec)\tDec. (asec)')
    print('W1-W2\t%0.6f\t%0.6f\t%0.6f\t%0.6f'%( (PositionsP[0][0] - PositionsP[1][0]), 
                                                (Positions[0][0] - Positions[1][0]) * d2a * np.cos(Positions[0][1] * np.pi/180.), 
                                                (PositionsP[0][1] - PositionsP[1][1]), 
                                                (Positions[0][1] - Positions[1][1]) * d2a ) )
    print('W1-W3\t%0.6f\t%0.6f\t%0.6f\t%0.6f'%( (PositionsP[0][0] - PositionsP[2][0]), 
                                                (Positions[0][0] - Positions[2][0]) * d2a * np.cos(Positions[0][1] * np.pi/180.), 
                                                (PositionsP[0][1] - PositionsP[2][1]), 
                                                (Positions[0][1] - Positions[2][1]) * d2a ) )
    print('W1-W4\t%0.6f\t%0.6f\t%0.6f\t%0.6f'%( (PositionsP[0][0] - PositionsP[3][0]), 
                                                (Positions[0][0] - Positions[3][0]) * d2a * np.cos(Positions[0][1] * np.pi/180.), 
                                                (PositionsP[0][1] - PositionsP[3][1]), 
                                                (Positions[0][1] - Positions[3][1]) * d2a ) )
    print('W2-W3\t%0.6f\t%0.6f\t%0.6f\t%0.6f'%( (PositionsP[1][0] - PositionsP[2][0]), 
                                                (Positions[1][0] - Positions[2][0]) * d2a * np.cos(Positions[1][1] * np.pi/180.), 
                                                (PositionsP[1][1] - PositionsP[2][1]), 
                                                (Positions[1][1] - Positions[2][1]) * d2a ) )
    print('W2-W4\t%0.6f\t%0.6f\t%0.6f\t%0.6f'%( (PositionsP[1][0] - PositionsP[3][0]), 
                                                (Positions[1][0] - Positions[3][0]) * d2a * np.cos(Positions[1][1] * np.pi/180.), 
                                                (PositionsP[1][1] - PositionsP[3][1]), 
                                                (Positions[1][1] - Positions[3][1]) * d2a ) )
    print('W3-W4\t%0.6f\t%0.6f\t%0.6f\t%0.6f\n'%( (PositionsP[2][0] - PositionsP[3][0]), 
                                                  (Positions[2][0] - Positions[3][0]) * d2a * np.cos(Positions[2][1] * np.pi/180.), 
                                                  (PositionsP[2][1] - PositionsP[3][1]), 
                                                  (Positions[2][1] - Positions[3][1]) * d2a ) )

  ####### Here is where we do the check against the parent population

  ## First the ellipticity part 
  t0   = Table.read('data/ellipticity_check_fixed.csv')

  j1   = np.where( (Amplitudes[0] >= t0['low']) & (Amplitudes[0] <= t0['high']) & (t0['band'] == 1) )[0]
  t1   = Table.read('data/1_%s_%s.hdf5'%(t0['low'][j1].data[0], t0['high'][j1].data[0]), path='data')
  S1   = t1['sources'].data
  #print 'W1 e survival probability:', np.interp(Eccent[0], np.arange(S1.size+1) / float(S1.size), np.concatenate([S1[::-1], S1[[0]]]) )

  j2   = np.where( (Amplitudes[1] >= t0['low']) & (Amplitudes[1] <= t0['high']) & (t0['band'] == 2) )
  t2   = Table.read('data/2_%s_%s.hdf5'%(t0['low'][j2].data[0], t0['high'][j2].data[0]), path='data')
  S2   = t2['sources'].data
  #print 'W2 e survival probability:', np.interp(Eccent[1], np.arange(S2.size+1) / float(S2.size), np.concatenate([S2[::-1], S2[[0]]]) )

  j3   = np.where( (Amplitudes[2] >= t0['low']) & (Amplitudes[2] <= t0['high']) & (t0['band'] == 3) )
  t3   = Table.read('data/3_%s_%s.hdf5'%(t0['low'][j3].data[0], t0['high'][j3].data[0]), path='data')
  S3   = t3['sources'].data
  #print 'W3 e survival probability:', np.interp(Eccent[2], np.arange(S3.size+1) / float(S3.size), np.concatenate([S3[::-1], S3[[0]]]) )

  j4   = np.where( (Amplitudes[3] >= t0['low']) & (Amplitudes[3] <= t0['high']) & (t0['band'] == 4) )
  t4   = Table.read('data/4_%s_%s.hdf5'%(t0['low'][j4].data[0], t0['high'][j4].data[0]), path='data')
  S4   = t4['sources'].data
  #print 'W4 e survival probability:', np.interp(Eccent[3], np.arange(S4.size+1) / float(S4.size), np.concatenate([S4[::-1], S4[[0]]]) )
  print('Ellipticity Survival Probabilities')
  print('W1\tW2\tW3\tW4')
  print('%0.3f\t%0.3f\t%0.3f\t%0.3f\n'%(np.interp(Eccent[0], np.arange(S1.size+1) / float(S1.size), np.concatenate([S1[::-1], S1[[0]]]) ),
                                        np.interp(Eccent[1], np.arange(S2.size+1) / float(S2.size), np.concatenate([S2[::-1], S2[[0]]]) ),
                                        np.interp(Eccent[2], np.arange(S3.size+1) / float(S3.size), np.concatenate([S3[::-1], S3[[0]]]) ),
                                        np.interp(Eccent[3], np.arange(S4.size+1) / float(S4.size), np.concatenate([S4[::-1], S4[[0]]]) )
                                       ) )


  ## Check the offsets:
  t1    = Table.read('data/Pixel_1_2_10.csv', format='ascii.ecsv')
  j     = np.where( (Amplitudes[1] >= t1['lowAmp']) & (Amplitudes[1] < t1['highAmp']) )
  T1std = util.FindStd(t1['semimajor'][j], t1['semiminor'][j], t1['meanx'][j], t1['meany'][j], t1['angle'][j], PositionsP[0][0] - PositionsP[1][0], PositionsP[0][1] - PositionsP[1][1])
  #print 'W1-W2 Std. Dev.:', T1std.data[0]

  t1    = Table.read('data/Pixel_1_3_6.csv', format='ascii.ecsv')
  j     = np.where( (Amplitudes[2] >= t1['lowAmp']) & (Amplitudes[2] < t1['highAmp']) )
  T2std = util.FindStd(t1['semimajor'][j], t1['semiminor'][j], t1['meanx'][j], t1['meany'][j], t1['angle'][j], PositionsP[0][0] - PositionsP[2][0], PositionsP[0][1] - PositionsP[2][1])
  #print 'W1-W3 Std. Dev.:', T2std.data[0]

  t1    = Table.read('data/Pixel_1_4_8.csv', format='ascii.ecsv')
  j     = np.where( (Amplitudes[3] >= t1['lowAmp']) & (Amplitudes[3] < t1['highAmp']) )
  T3std = util.FindStd(t1['semimajor'][j], t1['semiminor'][j], t1['meanx'][j], t1['meany'][j], t1['angle'][j], PositionsP[0][0] - PositionsP[3][0], PositionsP[0][1] - PositionsP[3][1])
  #print 'W1-W4 Std. Dev.:', T3std.data[0]

  t1    = Table.read('data/Pixel_2_3_6.csv', format='ascii.ecsv')
  j     = np.where( (Amplitudes[2] >= t1['lowAmp']) & (Amplitudes[2] < t1['highAmp']) )
  T4std = util.FindStd(t1['semimajor'][j], t1['semiminor'][j], t1['meanx'][j], t1['meany'][j], t1['angle'][j], PositionsP[1][0] - PositionsP[2][0], PositionsP[1][1] - PositionsP[2][1])
  #print 'W2-W3 Std. Dev.:', T4std.data[0]

  t1    = Table.read('data/Pixel_2_4_8.csv', format='ascii.ecsv')
  j     = np.where( (Amplitudes[3] >= t1['lowAmp']) & (Amplitudes[3] < t1['highAmp']) )
  T5std = util.FindStd(t1['semimajor'][j], t1['semiminor'][j], t1['meanx'][j], t1['meany'][j], t1['angle'][j], PositionsP[1][0] - PositionsP[3][0], PositionsP[1][1] - PositionsP[3][1])
  #print 'W2-W4 Std. Dev.:', T5std.data[0]

  t1    = Table.read('data/Pixel_3_4_8.csv', format='ascii.ecsv')
  j     = np.where( (Amplitudes[3] >= t1['lowAmp']) & (Amplitudes[3] < t1['highAmp']) )
  T6std = util.FindStd(t1['semimajor'][j], t1['semiminor'][j], t1['meanx'][j], t1['meany'][j], t1['angle'][j], PositionsP[2][0] - PositionsP[3][0], PositionsP[2][1] - PositionsP[3][1])
  #print 'W3-W4 Std. Dev.:', T6std.data[0]

  print('Offset Sigmas:')
  print('W1-W2\tW1-W3\tW1-W4\tW2-W3\tW2-W4\tW3-W4')
  print('%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n'%(T1std, T2std ,T3std ,T4std ,T5std ,T6std))


  if plot: 
    plt.show()


# Case for running in the command line
if __name__ == "__main__":

  # Parse the arguments
  argv = sys.argv
  try:
    ra = argv[1]
  except: 
    raise SyntaxError('python unwisest.py ra dec')
  try:
    dec = argv[2]
  except: 
    raise SyntaxError('python unwisest.py ra dec')

  # Run unwisest
  unwisest(ra=ra, dec=dec)

