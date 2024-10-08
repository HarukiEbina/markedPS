import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special, optimize, integrate, stats
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, interp1d, interp2d, BarycentricInterpolator
from time import time
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from timeit import timeit
from time import time
from copy import copy
from classy import Class
import sys
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
import scipy

import time
from scipy.special import hyp2f1, hyperu, gamma
import pyfftw

##################################################################################
# make plots prettier
import matplotlib
from matplotlib.pyplot import rc
import matplotlib.font_manager

# rc('font',**{'size':'22','family':'serif','serif':['CMU serif']})
rc('mathtext', **{'fontset':'cm'})
# rc('text', usetex=True)
# rc('legend',**{'fontsize':'18'})
rc('legend',**{'fontsize':'14'})

matplotlib.rcParams['axes.linewidth'] = 3
# matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 25 
matplotlib.rcParams['ytick.labelsize'] = 25
# matplotlib.rcParams['legend.fontsize'] = 25
matplotlib.rcParams['legend.fontsize'] = 14
#matplotlib.rcParams['legend.title_fontsize'] = 25
matplotlib.rcParams['xtick.major.size'] = 10
matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['xtick.minor.size'] = 5
matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['xtick.minor.width'] = 1.5
matplotlib.rcParams['ytick.minor.width'] = 1.5
matplotlib.rcParams['axes.titlesize'] = 30
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'