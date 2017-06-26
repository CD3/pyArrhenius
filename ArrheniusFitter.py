#! /usr/bin/python

import argparse, sys
import numpy as np
import mpmath  as mp
from scipy.stats import linregress
from scipy.optimize import brentq,minimize,minimize_scalar
import pint

try:
  from pathos.multiprocessing import ProcessingPool as Pool
  have_pool = 1
except:
  have_pool = 0

from ArrheniusIntegral import ArrheniusIntegral,LoadThermalProfile,ComputeThreshold

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

def ComputeScalingFactors(Profiles,Tbs,A,Ea):
  SFs = dict()
  for profile in Profiles:
    Tb = Tbs[file] if isinstance(Tbs,dict) else Tbs
    if isinstance( profile, (str, unicode) ):
      t,T = LoadThermalProfile( profile, Tb )
    else:
      t,T = profile

    SFs[profile] = ComputeThreshold(t,T,A,Ea)
    

  return SFs

def ComputeScalingFactorsRSquared(Profiles,Tbs,A,Ea):
  SFs = ComputeScalingFactors(Profiles,Tbs,A,Ea)
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
    i = np.argmax(T[::-1])
    tau = t[i]
    Tmax = T[i]

    x.append(1/Tmax)
    y.append(float(mp.log(tau)))


  m,b,r,p,err = linregress(x,y)

  Ea = mp.mpmathify(R*m)
  A = mp.exp(-b)

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

  Ea = mp.mpmathify(R*m)
  A = mp.exp(-b)

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

  Ea = mp.mpmathify(xavg)
  # log(A) = yavg
  # A = exp(yavg)
  A  = mp.exp(yavg)

  return A,Ea

def MinimizeLogAStdDevWithLinearRegressionMethod(config):
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

def MinimizeLogAStdDevAndScalingFactorsWithLinearRegressionMethod(config):
  A,Ea = MinimizeLogAStdDevWithLinearRegressionMethod(config)
  # now find the value for log(A) that minimizes the scaling factor err
  regressions = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    regressions[file] = ComputeLogAvsEaLine(t,T,[config.Ea_min,config.Ea_max])
  logAs = [regressions[f]['m']*Ea + regressions[f]['b'] for f in config.files]
  def cost(logA):
    err = ComputeScalingFactorsRSquared(config.files, config.baseline_temperature, mp.exp(logA), Ea )
    return err
  res = minimize_scalar( cost, bounds=(min(logAs),max(logAs)), method='bounded' )
  logA = res.x

  A = mp.exp(logA)
  return A,Ea

def MinimizeLogAStdDevAndScalingFactorsMethod(config):
  profiles = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    profiles[file] = [t,T]

  # first find Ea that minimized stddev for logA
  def cost(Ea):
    logAs = [ ComputeLogA(profiles[f][0],profiles[f][1], Ea) for f in profiles.keys()]
    return np.std(logAs)

  res = minimize_scalar( cost, bounds=(config.Ea_min,config.Ea_max), method='bounded' )
  Ea = res.x
  logAs = [ ComputeLogA(profiles[f][0],profiles[f][1], Ea) for f in profiles.keys()]

  def cost(logA):
    err = ComputeScalingFactorsRSquared(config.files, config.baseline_temperature, mp.exp(logA), Ea )
    return err

  res = minimize_scalar( cost, bounds=(min(logAs),max(logAs)), method='bounded' )
  logA = res.x

  A = mp.exp(logA)
  return A,Ea


def ScanForMinimumScalingFactorsMethod(config):
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
      err = ComputeScalingFactorsRSquared(config.files, config.baseline_temperature, mp.exp(logA), x )
      return err
    res = minimize_scalar( f, bounds=(min(ys),max(ys)), method='bounded' )
    y = res.x
    e = ComputeScalingFactorsRSquared(config.files, config.baseline_temperature, mp.exp(logA), x )
    if err is None or e < err:
      err = e
      Ea = x
      logA = y

    printProgressBar(i,len(Eas),prefix='Progress:', suffix='Complete', length=50)

  return exp(logA),Ea

def MinimizeScalingFactorsMethod(config):
  profiles = dict()
  for file in config.files:
    t,T = LoadThermalProfile( file, config.baseline_temperature )
    profiles[file] = [t,T]

  class Cost:
    thermal_profiles = profiles
    baseline_temperature = config.baseline_temperature

    def __call__( self, x ):
      A = x[0]
      Ea = x[1]
      err = ComputeScalingFactorsRSquared(self.thermal_profiles, self.baseline_temperature, A, Ea )
      return err

  def Process(Ea):
    logAs = [ ComputeLogA(cost.thermal_profiles[f][0],cost.thermal_profiles[f][1], Ea) for f in cost.thermal_profiles.keys() ]
    for A in np.logspace(float(min(logAs)), float(max(logAs)), 50):
      print A, Ea, cost( [A,Ea] )
  cost = Cost()

  # pool = Pool(config.num_jobs)
  # pool.map_async(Process, np.logspace(np.log10(args.Ea_min), np.log10(args.Ea_max), 500)

  for Ea in np.logspace(np.log10(args.Ea_min), np.log10(args.Ea_max), 500):
    Process(Ea)



  x0 = MinimizeLogAStdDevWithLinearRegressionMethod(config)

  bounds = [ (None,None), (config.Ea_min,config.Ea_max) ]
  constraints = [ { 'type' : 'ineq', 'fun' : logA_ll }
                , { 'type' : 'ineq', 'fun' : logA_ul } ]

  res = minimize(cost, (x0[0],x0[1]-1e5), bounds=bounds, constraints=constraints,options={'eps':1e5})
  print res
  print res.x
  




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Find a set of Arrhenius coefficients (A and Ea) that best fit a given set of threshold thermal profiles.")
  parser.add_argument("--Ea-min", type=np.float64, default=1e3, help="minimum Ea value to consider when searching for the best fit.")
  parser.add_argument("--Ea-max", type=np.float64, default=1e7, help="maximum Ea value to consider when searching for the best fit.")
  parser.add_argument("--baseline-temperature","-T0", type=np.float64, default=0,  help="baseline temperature to all to thermal profiles.")
  parser.add_argument("--methods",default='all', help="a list of methods to use.")
  parser.add_argument("--list-methods",action='store_true', default=False, help="list the available methods and exit.")
  parser.add_argument("--logAvsEa",action='store_true', default=False, help="Calculate logA vs Ea and write it to file..")
  parser.add_argument("--logAvsEa-N", type=int, default=100, help="the number of points to compute log(A) vs Ea at.")
  parser.add_argument("--no-fit",action='store_true',default=False, help="Do not perform an A,Ea fit.")
  parser.add_argument("--num-jobs",type=int,default=4, help="Number of jobs (processes) to use when doing parallel processing.")
  parser.add_argument("files", metavar="FILE", nargs="*", help="Files containing threshold thermal profile data.")
  args = parser.parse_args() 

  methods = { "constant temperature linear regression" : LinearRegressionMethod
            , "effective exposure linear regression" : EffectiveExposureMethod
            , "minimize log(A) standard deviation with linear regression" : MinimizeLogAStdDevWithLinearRegressionMethod
            , "minimize log(A) standard deviation and scaling factors with linear regression" : MinimizeLogAStdDevAndScalingFactorsWithLinearRegressionMethod
            # , "average line intersection" : AverageLineIntersectionMethod
            # , "scan for minimum scaling factors" : ScanForMinimumScalingFactorsMethod
            # , "minimize scaling factors" : MinimizeScalingFactorsMethod
            , "minimize log(A) standard deviation and scaling factors" : MinimizeLogAStdDevAndScalingFactorsMethod
            }

  if args.list_methods:
    for m in methods:
      print m
    sys.exit(0)

  if args.logAvsEa:
    N = args.logAvsEa_N
    for file in args.files:
      fn = "{file}.logAvsEa.txt".format(file=file)
      print "Generating {fn}".format(fn=fn)
      t,T = LoadThermalProfile( file, args.baseline_temperature )

      Ea = np.zeros(N,dtype='float64')
      logA = np.zeros(N,dtype='float64')
      
      Eas = np.logspace(np.log10(args.Ea_min), np.log10(args.Ea_max), args.logAvsEa_N)
      Eas = np.linspace(args.Ea_min, args.Ea_max, num=args.logAvsEa_N)
      for i in range(len(Eas)):
        Ea[i] = Eas[i]
        logA[i] = ComputeLogA(t,T,Eas[i])
      
      np.savetxt(fn, zip(Ea,logA))
        

  if not args.no_fit:

    if args.methods == 'all':
      args.methods = ','.join(methods.keys())

    for method in args.methods.split(','):
      print "running",method,"method"
      A,Ea = methods[method](args)
      print " A: {0} 1/s".format(mp.nstr(A,3))
      print "Ea: {0} J/mol".format(mp.nstr(Ea,3))
      SFs = ComputeScalingFactors(args.files,args.baseline_temperature,A,Ea)
      err = 0
      for file in SFs:
        print "{file}: {threshold}".format(file=file, threshold=SFs[file])
        err += (SFs[file] - 1)**2
      print "R^2: {err}".format(err=err)
      print


