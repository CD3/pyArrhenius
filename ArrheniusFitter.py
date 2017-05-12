#! /usr/bin/python

import argparse, sys
import numpy as np
import mpmath  as mp
from scipy.stats import linregress
from scipy.optimize import brentq,minimize_scalar
import matplotlib.pyplot as plt
import pint

from ArrheniusIntegral import ArrheniusIntegral,LoadThermalProfile

np.seterr(all='raise')

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

R = Q_(8.314, 'J/mol/degK').magnitude

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = 'x'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stderr.write('\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix))
    sys.stderr.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

def ComputeLogA(t,T,Ea):
  '''
  Calculates the frequency factor for a given activation energy that will give
  Omega = 1. log(A) (natural log) is actually returned.
  '''
  sum = ArrheniusIntegral(t,T,1,Ea)
  # 1 = A*sum
  # 1/A = sum
  # log(1/A) = log(sum) = log(1) - log(A)
  # log(A) = -log(sum)
  if sum == 0:
    return float('inf')
  return -mp.log(sum)

def ComputeScalingFactors(files,Tbs,Ea,A):
  SFs = dict()
  for file in files:
    Tb = Tbs[file] if isinstance(Tbs,dict) else Tbs
    t,T = LoadThermalProfile( file, Tb )
    T0 = T[0]
    dT = T - T0

    min = 0
    max = 5
    x0 = None
    while x0 is None and min < 5e10:
      try:
        f = lambda x : mp.log(A*ArrheniusIntegral(t,x*dT+T0,1,Ea))
        x0,r = brentq( f, min, max, full_output=True)
      except ValueError as e:
        min = max
        max *= 100
        pass

    SFs[file] = x0 if x0 is not None else max

  return SFs

def ComputeScalingFactorsRSquared(files,Tbs,Ea,A):
  SFs = ComputeScalingFactors(files,Tbs,Ea,A)
  sum = 0
  for f in SFs:
    sum += (SFs[f] - 1)**2
  return sum

def ComputeLogAvsEaLine(t,T,domain,N=10):
  Eas = np.logspace( np.log10(domain[0]), np.log10(domain[1]), num=N )
  x = []
  y = []
  for Ea in Eas:
    logA = ComputeLogA(t,T,Ea)
    x.append(Ea)
    y.append(float(logA))

  x = np.array(x)
  y = np.array(y)

  m,b,r,p,err = linregress(x,y)
  regression = { 'm' : m
               , 'b' : b
               , 'r' : r
               , 'p' : p
               , 'err' : err
               }

  return  regression




def LinearRegressionMethod(config):
  '''Fit a line to ln(tau) = Ea/R 1/T - ln(A).'''
  x = list()
  y = list()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    i = np.argmax(T)
    tau = t[i]
    Tmax = T[i]

    x.append(1/Tmax)
    y.append(float(mp.log(tau)))


  m,b,r,p,err = linregress(x,y)

  Ea = np.float64(R*m)
  A = np.float64(mp.exp(-b))

  return A,Ea

def EffectiveExposureMethod(config):
  regressions = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    regressions[file] = ComputeLogAvsEaLine(t,T,[config.Ea_min,config.Ea_max])

  x = list()
  y = list()
  for f in regressions:
    # We want to fit a line to ln(teff) = Ea / R*Teff - ln(A)
    # We have already fit ln(A) = m*Ea + b
    # which, for a constant temperature exposure would be
    # ln(A) = (1/R*Teff)*Ea - ln(teff)
    # with
    # m = 1 / R*Teff -> 1/Teff = m*R
    # b = -log(teff)
    m = regressions[f]['m']
    b = regressions[f]['b']

    x.append( R*m )
    y.append( -b )
    
  m,b,r,p,err = linregress(x,y)

  Ea = R*m
  A = np.float64(mp.exp(-b))

  return A,Ea

def AverageLineIntersectionMethod(config):
  regressions = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    regressions[file] = ComputeLogAvsEaLine(t,T,[config.Ea_min,config.Ea_max])

  intersections = []
  xsum = np.float64(0)
  ysum = np.float64(0)
  for f1 in regressions:
    m1 = regressions[f1]['m']
    b1 = regressions[f1]['b']
    for f2 in regressions:
      if f2 is f1:
        continue
      m2 = regressions[f2]['m']
      b2 = regressions[f2]['b']

      x = (b2 - b1)/(m1 - m2)
      y1 = m1*x+b1
      y2 = m2*x+b2
      if y2 - y1 > 1e-13:
        print "WARNING: intersections don't agree."
        print x,y1,y2,y2-y1
        print "Taking average."
      y = (y1+y2)/2
      intersections.append( (x,y1) )

      xsum += x
      ysum += y

  xavg = xsum/len(intersections)
  yavg = ysum/len(intersections)

  Ea = xavg
  # log(A) = yavg
  # A = exp(yavg)
  A  = np.float64(mp.exp(yavg))

  return A,Ea

def MinimizeLogAStdDevMethod(config):
  regressions = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    regressions[file] = ComputeLogAvsEaLine(t,T,[config.Ea_min,config.Ea_max])

  def f(Ea):
    logAs = [regressions[f]['m']*Ea + regressions[f]['b'] for f in config.files ]
    return np.std(logAs)

  res = minimize_scalar(f, bounds=(config.Ea_min,config.Ea_max), method='bounded')
  Ea = mp.mpmathify(res.x)
  logA = np.mean([regressions[f]['m']*Ea + regressions[f]['b'] for f in config.files])
  A = mp.exp(logA)
  return A,Ea

def MinimizeLogAStdDevAndScalingFactorsMethod(config):
  # use log(A) minimization to get value for Ea
  A,Ea = MinimizeLogAStdDevMethod(config)
  # now find the value for log(A) that minimizes the scaling factor err
  regressions = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    regressions[file] = ComputeLogAvsEaLine(t,T,[config.Ea_min,config.Ea_max])
  logAs = [regressions[f]['m']*Ea + regressions[f]['b'] for f in config.files]
  def f(logA):
    err = ComputeScalingFactorsRSquared(config.files, config.baseline_temperature, Ea, mp.exp(logA) )
    return err
  res = minimize_scalar( f, bounds=(min(logAs),max(logAs)), method='bounded' )
  logA = res.x

  A = mp.exp(logA)
  return A,Ea

def MinimizeScalingFactorsMethod(config):
  regressions = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    regressions[file] = ComputeLogAvsEaLine(t,T,[config.Ea_min,config.Ea_max])


  logA = None
  Ea = None
  err = None
  Eas = np.logspace( np.log10(config.Ea_min), np.log10(config.Ea_max), num=100 )
  printProgressBar(0,len(Eas),prefix='Progress:', suffix='Complete', length=50)
  i = 0
  for x in Eas:
    i += 1
    ys = list()
    for f in regressions:
      m = regressions[f]['m']
      b = regressions[f]['b']
      ys.append(m*x+b)

    def f(logA):
      err = ComputeScalingFactorsRSquared(config.files, config.baseline_temperature, x, mp.exp(logA) )
      return err
    res = minimize_scalar( f, bounds=(min(ys),max(ys)), method='bounded' )
    y = res.x
    e = ComputeScalingFactorsRSquared(config.files, config.baseline_temperature, x, mp.exp(logA) )
    if err is None or e < err:
      err = e
      Ea = x
      logA = y

    printProgressBar(i,len(Eas),prefix='Progress:', suffix='Complete', length=50)

  return exp(logA),Ea







if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Find a set of Arrhenius coefficients (A and Ea) that best fit a given set of threshold thermal profiles.")
  parser.add_argument("--Ea-min", type=np.float64, default=1e4, help="minimum Ea value to compute.")
  parser.add_argument("--Ea-max", type=np.float64, default=1e8, help="maximum Ea value to compute.")
  parser.add_argument("--baseline-temperature","-T0", type=np.float64, default=310,  help="number of Ea values to compute.")
  parser.add_argument("--methods",default='all', help="list of methods to use.")
  parser.add_argument("files", metavar="FILE", nargs="*", help="Files containing threshold thermal profile data.")
  args = parser.parse_args() 

  methods = { "constant temperature linear regression" : LinearRegressionMethod
            , "effective exposure linear regression" : EffectiveExposureMethod
            , "average line intersection" : AverageLineIntersectionMethod
            , "minimize log(A) standard deviation" : MinimizeLogAStdDevMethod
            , "minimize log(A) standard deviation and scaling factors" : MinimizeLogAStdDevAndScalingFactorsMethod
            # , "minimize scaling factors" : MinimizeScalingFactorsMethod
            }

  if args.methods == 'all':
    args.methods = ','.join(methods.keys())

  for method in args.methods.split(','):
    print "running",method,"method"
    A,Ea = methods[method](args)
    print " A: {0} 1/s".format(mp.nstr(A,3))
    print "Ea: {0} J/mol".format(mp.nstr(Ea,3))
    SFs = ComputeScalingFactors(args.files,args.baseline_temperature,Ea,A)
    err = 0
    for file in SFs:
      print "{file}: {threshold}".format(file=file, threshold=SFs[file])
      err += (SFs[file] - 1)**2
    print "R^2: {err}".format(err=err)
    print

