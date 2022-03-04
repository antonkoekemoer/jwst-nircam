#!/usr/bin/env python
#
# fix_settling_shifts.py
#
# Author: Anton M. Koekemoer, STScI, February 2022
#
# Script to fix shifts between groups introduced by settling issue.
#
# Input arguments:
#   required:  image filename (single "uncal.fits" file)
#   optional:  boxpeaksize (box size for searching central peak; default=20 pixels)
#   optional:  boxfitsize (box size for fitting central peak; default=200 pixels)
#
# Output: Level-2 calibrated file ("cal.fits" file)
#
#
# Example of how to run it - from the unix command line:
#
#  $ fix_settling_shifts.py   jw01143001001_02101_00001_nrcalong_uncal.fits
#
# which produces:
#                             jw01143001001_02101_00001_nrcalong_cal.fits



import argparse, os, sys
from glob import glob

import astropy
from astropy.io import ascii as asc
from astropy.io import fits
from astropy import stats
from astropy import nddata
from astropy.nddata import block_reduce
from astropy.modeling import models, fitting

import jwst
from jwst.pipeline import calwebb_detector1
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Image2Pipeline
from jwst.jump import JumpStep
from jwst.ramp_fitting import RampFitStep
from jwst.gain_scale import GainScaleStep

import scipy
from scipy.ndimage import gaussian_filter, median_filter
from scipy import signal

import numpy as np



def crosscorfit(data_template,data_to_be_shifted,corrboxpeak,corrboxfit):
  #
  corr = scipy.signal.fftconvolve(data_template,data_to_be_shifted[::-1,::-1],mode='same')
  #
  xcen,ycen = int(float(corr.shape[0])/2.),int(float(corr.shape[1])/2.)
  #
  box2 = int(float(corrboxpeak)/2.)
  #
  ypeak,xpeak = np.unravel_index(np.argmax(corr[ycen-box2:ycen+box2,xcen-box2:xcen+box2]),(corrboxpeak,corrboxpeak))
  #
  x0,y0 = xcen-box2+xpeak,ycen-box2+ypeak
  #
  ampl = corr[y0,x0]
  #
  x_stddev0,y_stddev0 = 2.,2.
  #
  corrboxfit2 = float(corrboxfit)/2.
  #
  x1,y1 = xcen-int(corrboxfit2),ycen-int(corrboxfit2)
  x2,y2 = x1+corrboxfit,y1+corrboxfit
  #
  corr_fit = corr[y1:y2,x1:x2]
  #
  x0fit,y0fit = x0-x1,y0-y1
  #
  y_array_2d,x_array_2d = np.mgrid[:corrboxfit,:corrboxfit]
  #
  gfit_init = models.Gaussian2D(amplitude=ampl,x_mean=x0fit,y_mean=y0fit,x_stddev=x_stddev0,y_stddev=y_stddev0,theta=0.)
  gfit_init.theta.fixed = True
  #
  gfit_model   = fitting.LevMarLSQFitter()
  gfit_results = gfit_model(gfit_init,x_array_2d,y_array_2d,corr_fit)
  #
  amplfit = gfit_results.amplitude.value
  thetafit = gfit_results.theta.value
  xfit,yfit = gfit_results.x_mean.value,gfit_results.y_mean.value
  xsigma,ysigma = gfit_results.x_stddev.value,gfit_results.y_stddev.value
  dx,dy = xfit-corrboxfit2,yfit-corrboxfit2
  #
  print('fit1  results:  %s  %3i %3i %5i %5i %8.3f %8.3f %8.3f %8.3f %14.1f %14.1f %8.3f %8.3f %8.3f' % (uncalfile,ni+1,ng+1,x0fit,y0fit,xfit,yfit,dx,dy,ampl,amplfit,thetafit,xsigma,ysigma))
  #
  #
  sigma_new = np.min([xsigma,ysigma])
  #
  gfit_init2 = models.Gaussian2D(amplitude=ampl,x_mean=x0fit,y_mean=y0fit,x_stddev=sigma_new,y_stddev=sigma_new,theta=0.)
  gfit_init2.x_stddev.fixed = True
  gfit_init2.y_stddev.fixed = True
  gfit_init2.theta.fixed = True
  #
  gfit_model2   = fitting.LevMarLSQFitter()
  gfit_results2 = gfit_model2(gfit_init2,x_array_2d,y_array_2d,corr_fit)
  #
  amplfit = gfit_results2.amplitude.value
  thetafit = gfit_results2.theta.value
  xfit,yfit = gfit_results2.x_mean.value,gfit_results2.y_mean.value
  xsigma,ysigma = gfit_results2.x_stddev.value,gfit_results2.y_stddev.value
  dx,dy = xfit-corrboxfit2,yfit-corrboxfit2
  #
  print('fit2  results:  %s  %3i %3i %5i %5i %8.3f %8.3f %8.3f %8.3f %14.1f %14.1f %8.3f %8.3f %8.3f' % (uncalfile,ni+1,ng+1,x0fit,y0fit,xfit,yfit,dx,dy,ampl,amplfit,thetafit,xsigma,ysigma))
  #
  if ((xpeak in [0,corrboxpeak]) or (ypeak in [0,corrboxpeak])): exit_need_adjustment('xpeak, ypeak = '+repr(xpeak)+', '+repr(ypeak)+';  corrboxpeak = '+repr(corrboxpeak))
  if ((xsigma > box2) or (ysigma > box2)): exit_need_adjustment('xsigma, ysigma = '+repr(xsigma)+', '+repr(ysigma)+';  box2 = '+repr(box2))
  #
  return dx,dy



def apply_shift(data_to_be_shifted,dx,dy,bkgd):
  #
  dxi,dyi = round(dx),round(dy)
  #
  ny,nx = data_to_be_shifted.shape
  #
  if (dxi >= 0):
    x1old,x2old = 0,nx-dxi
    x1new,x2new = dxi,nx
  else:
    x1old,x2old = -dxi,nx
    x1new,x2new = 0,nx+dxi
  #
  if (dyi >= 0):
    y1old,y2old = 0,ny-dyi
    y1new,y2new = dyi,ny
  else:
    y1old,y2old = -dyi,ny
    y1new,y2new = 0,ny+dyi
  #
  data_to_be_shifted_new = np.full(data_to_be_shifted.shape,bkgd,dtype=np.float32)
  data_to_be_shifted_new[y1new:y2new,x1new:x2new] = data_to_be_shifted[y1old:y2old,x1old:x2old]
  #
  return data_to_be_shifted_new



def exit_need_adjustment(err_info):
  #
  print('''
  ***
  *** ERROR - The code has encountered potentially problematic results, and is therefore exiting.
  ***''')
  print('  *** Error info is:  ',err_info)
  print('''  ***
  ***
  *** Please contact the author (Anton Koekemoer) to discuss how to run it on this dataset.
  ***
  ''')
  sys.exit()



if __name__ == '__main__':
  #
  print('''
  *** -----------------------------------------------------------------------------------
  ***
  *** fix_settling_shifts.py
  ***
  *** Author: Anton M. Koekemoer, STScI, February 2022
  ***
  *** Script to fix shifts between groups introduced by settling issue.
  ***
  *** Input arguments:
  ***	required:  image filename (single "uncal.fits" file)
  ***	optional:  boxpeaksize (box size for searching central peak; default=20 pixels)
  ***	optional:  boxfitsize (box size for fitting central peak; default=200 pixels)
  ***
  *** Output: Level-2 calibrated file ("cal.fits" file)
  ***
  ***
  *** Example of how to run it - from the unix command line:
  ***
  ***  $ fix_settling_shifts.py   jw01143001001_02101_00001_nrcalong_uncal.fits
  ***
  *** which produces:
  ***                             jw01143001001_02101_00001_nrcalong_cal.fits
  ***
  *** -----------------------------------------------------------------------------------
  ''')
  #
  parser = argparse.ArgumentParser(description='Fix shifts between groups introduced by settling issue.')
  parser.add_argument('image', default='NONE', type=str, help='Input uncal.fits filename')
  parser.add_argument('-bp','--boxpeaksize',default=20, type=int, help='Box size for searching central peak') 
  parser.add_argument('-bf','--boxfitsize',default=200, type=int, help='Box size for fitting central peak')
  #
  options = parser.parse_args()
  uncalfile   = options.image
  corrboxpeak = options.boxpeaksize
  corrboxfit  = options.boxfitsize
  #
  parameter_dict = {'jump':       {'skip': True},
                    'ramp_fit':   {'skip': True},
                    'gain_scale': {'skip': True}}
  #
  rampfile = uncalfile.replace('_uncal.fits', '_ramp.fits')
  #
  if (not os.path.exists(rampfile)):
    rampdata = calwebb_detector1.Detector1Pipeline.call(uncalfile, steps=parameter_dict, save_results=True)
  #
  hdr0 = fits.getheader(uncalfile,0)
  n_ints         = hdr0['NINTS']
  n_grps_per_int = hdr0['NGROUPS']
  #
  data = fits.getdata(uncalfile,1)
  #
  ramp_cube_aligned = np.zeros(data.shape).astype(np.float32)
  #
  first_group = True


  for ni in range(n_ints):
    #
    for ng in range(n_grps_per_int):
      #
      if (ng != 0): data_intgrp_prev = data_intgrp
      #
      data_intgrp = fits.getdata(rampfile,1)[ni,ng,:,:]
      #
      if (ng == 0):
        #
        data_diff = data_intgrp
        #
      else:
        #
        data_diff = data_intgrp - data_intgrp_prev
      #
      mean,med,rms = stats.sigma_clipped_stats(data_diff, maxiters=10, sigma_lower=6., sigma_upper=4)
      #
      print('n_int = ',ni+1,';  n_grp = ',ng+1,';  mean, med, rms = ',mean,med,rms)
      #
      data_diff_med = scipy.ndimage.median_filter(input=data_diff, size=3, mode='constant', cval=0.)
      #
      data_diff_sub = data_diff_med - med
      #
      data_diff_sub[np.where(data_diff_sub < (5.*rms))] = 0.
      #
      data_diff_sub_gauss = gaussian_filter(data_diff_sub, sigma=1.0, truncate=5., order=0, mode='constant', cval=0.).astype(np.float32)
      #
      if (first_group):
        #
        first_group = False
        #
        group_template = data_diff_sub_gauss
        #
        ramp_cube_aligned[0,0,:,:] = data_diff
        #
      else:
        #
        dx,dy = crosscorfit(group_template,data_diff_sub_gauss,corrboxpeak,corrboxfit)
        #
        data_diff_sub_gauss_shifted = apply_shift(data_diff_sub_gauss,dx,dy,0.)
        #
        dx_check,dy_check = crosscorfit(group_template,data_diff_sub_gauss_shifted,corrboxpeak,corrboxfit)
        #
        print('shift results:  %s  %3i %3i %5i %5i %8.3f %8.3f %8.3f %8.3f' % (uncalfile,ni+1,ng+1,round(dx),round(dy),dx,dy,dx_check,dy_check))
        #
        data_diff_shifted = apply_shift(data_diff,dx,dy,med)
        #
        if (ng == 0):
          #
          ramp_cube_aligned[ni,ng,:,:] = data_diff_shifted
          #
        else:
          #
          ramp_cube_aligned[ni,ng,:,:] = ramp_cube_aligned[ni,ng-1,:,:] + data_diff_shifted


  rampfile_aligned = rampfile[:-10] + '_aligned_ramp.fits'
  #
  a = os.system('/bin/rm -f '+rampfile_aligned)
  a = os.system('/bin/cp -p '+rampfile+' '+rampfile_aligned)
  #
  f = fits.open(rampfile_aligned,'update')
  #
  f[1].data = ramp_cube_aligned
  #
  f.flush()
  f.close()
  #
  caldetector1_output_jumpstep   = JumpStep.call(rampfile_aligned,                    save_results=False)
  caldetector1_output_rampfit    = RampFitStep.call(caldetector1_output_jumpstep,     save_results=False)
  caldetector1_output_gainscale0 = GainScaleStep.call(caldetector1_output_rampfit[0], save_results=False)
  caldetector1_output_gainscale1 = GainScaleStep.call(caldetector1_output_rampfit[1], save_results=False)
  #
  calimage2_output0 = Image2Pipeline.call(caldetector1_output_gainscale0, save_results=True)
  calimage2_output1 = Image2Pipeline.call(caldetector1_output_gainscale1, save_results=True)


