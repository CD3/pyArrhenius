#! /usr/bin/python
import mpmath as mp
import numpy as np

def ArrheniusIntegral(t,T,A,Ea):
  '''
  Calculates the Arrhenius integral for a time-dependent temperature profile for a
  given set of A and Ea coefficients.

  @param t [list-like] times (in seconds)
  @param T [list-like] temperatures (in K)
  @param A [number] frequency factor (in 1/s)
  @param Ea [number] activation energy (in J/mol)
  @return omega [number] the damage parameter (dimensionless)
  '''
  sum = 0
  for i in range(len(T)-1):
    sum = sum + (mp.exp( -Ea / (8.314 * T[i])) + mp.exp( -Ea  / (8.314 * T[i+1])))*(t[i+1]-t[i])
  sum = A*sum*0.5

  return sum

def LoadThermalProfile(file,T0=0.0):
  t,T = np.loadtxt( file, unpack=True, dtype='float64' )
  T = T + T0
  return t,T

if __name__ == "__main__":
  import argparse, sys
  parser = argparse.ArgumentParser(description="Compute the damage integral (Arrhenius) for one or more temperature profiles.")
  parser.add_argument("--temperature-offset","-T0", type=np.float64, default=0,  help="An otfset temperature that will be added to all temperatures before integrating.")
  parser.add_argument("--activation-energy","-Ea", type=np.float64, default=6.28e5,  help="The activation energy to use [J/mol].")
  parser.add_argument("--frequency-factor","-A", type=np.float64, default=3.1e99,  help="The activation energy to use [J/mol].")
  parser.add_argument("files", metavar="FILE", nargs="*", help="Files containing temperature profile data.")
  args = parser.parse_args() 

  for file in args.files:
    t,T = LoadThermalProfile(file,args.temperature_offset)
    Omega = ArrheniusIntegral(t,T,args.frequency_factor,args.activation_energy)
    print "{file}: {Omega}".format(file=file,Omega=Omega)


