#! /usr/bin/python
import mpmath as mp
import numpy as np
from scipy.optimize import brentq,minimize,minimize_scalar
import multiprocessing as mproc

def ArrheniusIntegralTrapezoid(t,T,A,Ea):
  '''
  Calculates the Arrhenius integral for a time-dependent temperature profile for a
  given set of A and Ea coefficients using the trapezoid method.

  @param t [list-like] times (in seconds)
  @param T [list-like] temperatures (in K)
  @param A [number] frequency factor (in 1/s)
  @param Ea [number] activation energy (in J/mol)
  @return omega [number] the damage parameter (dimensionless)
  '''
  sum = 0
  alpha = -Ea/8.314
  exp_last = mp.exp( alpha/T[0])
  N = len(T)
  for i in range(1,N):
    exp_now = mp.exp( alpha/T[i])
    sum = sum + (exp_now + exp_last)*(t[i]-t[i-1])
    exp_last = exp_now
  sum = A*sum*0.5

  return sum

def ArrheniusIntegralLinearTempQuadrature(t,T,A,Ea):
  '''
  Calculates the Arrhenius integral for a time-dependent temperature profile for a
  given set of A and Ea coefficients using a semi-analytical quadrature. Rather
  than assuming the damge rate is linear between time poitns, it assumes that the temperature
  is linear between points.

  @param t [list-like] times (in seconds)
  @param T [list-like] temperatures (in K)
  @param A [number] frequency factor (in 1/s)
  @param Ea [number] activation energy (in J/mol)
  @return omega [number] the damage parameter (dimensionless)
  '''


  sum = 0
  E2ox_last = mp.expint(2,T[0])/T[0]
  N = len(T)
  for i in range(1,N-1):
    E2ox_now = mp.expint(2,T[i])/T[i]
    dT = T[i] - T[i-1]
    if abs(dT) > 0:
      _1om = (t[i] - t[i-1])/dT
      sum = sum + _1om * ( E2ox_last - E2ox_now )
    E2ox_last = E2ox_now
  sum = sum*A*Ea/8.314

  return sum


ArrheniusIntegral = ArrheniusIntegralTrapezoid
# ArrheniusIntegral = ArrheniusIntegralLinearTempQuadrature

def ComputeThreshold(t,T,A,Ea):
  T0 = T[0]
  dT = T - T0

  min = 0
  max = 5
  x0 = None
  while x0 is None and min < 5e10:
    try:
      # f = lambda x : mp.log(ArrheniusIntegral(t,x*dT+T0,A,Ea))
      f = lambda x : ArrheniusIntegral(t,x*dT+T0,A,Ea) - 1
      x0,r = brentq( f, min, max, full_output=True)
    except ValueError as e:
      min = max
      max *= 100
      pass

  return x0 if x0 is not None else max


def LoadThermalProfile(file,T0=0.0):
  t,T = np.loadtxt( file, unpack=True, dtype='float64' )
  T = T + T0
  return t,T

def AnalyseFile(file):
  t,T = LoadThermalProfile(file,args.temperature_offset)
  Omega = ArrheniusIntegral(t,T,args.frequency_factor,args.activation_energy)
  thresh = ComputeThreshold(t,T,args.frequency_factor,args.activation_energy)
  return "{file} | {Omega} | {thresh}".format(file=file,Omega=Omega,thresh=thresh)

if __name__ == "__main__":
  import argparse, sys
  parser = argparse.ArgumentParser(description="Compute the damage integral (Arrhenius) for one or more temperature profiles.")
  parser.add_argument("--temperature-offset","-T0", type=np.float64, default=0,  help="An otfset temperature that will be added to all temperatures before integrating.")
  parser.add_argument("--activation-energy","-Ea", type=np.float64, default=6.28e5,  help="The activation energy to use [J/mol].")
  parser.add_argument("--frequency-factor","-A", type=np.float64, default=3.1e99,  help="The activation energy to use [J/mol].")
  parser.add_argument("--num-jobs","-j", type=int, default=1,  help="Number of jobs (processes) to use when analyzing multiple files.")
  parser.add_argument("files", metavar="FILE", nargs="*", help="Files containing temperature profile data.")
  args = parser.parse_args() 

  pool = mproc.Pool(processes=args.num_jobs)


  print "filename | Omega | theshold"
  for output in pool.imap_unordered(AnalyseFile, args.files):
    print output


