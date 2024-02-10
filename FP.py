import cmath
from matplotlib import mlab
from scipy.integrate import odeint
import matplotlib.pylab as mp
import numpy as np

n0 = 2.82 # outer medium index
nL = 1.5 # low index medium
nH = 2.82 # high index medium
n3 = nH # cavity

h1 = 1.5e-6/(4*nL)
h2 = 1.5e-6/(4*nH)
hc = h1+h2
print(hc)
h3 = 0.1e-6
hcavity = np.arange(0.0,3*hc,hc/100)

wavelength = np.arange(500,3001,0.5)*1e-9

structure = [0,1,2,1,2,1,2,1,2,1,2,3,1,2,1,2,1,2,1,2,0]
param = {0:[n0,0],1:[nL,h1],2:[nH,h2],3:[n3,h3]}

Pmin=[]
Pout =[]
for hh in range(0,len(hcavity)):
    param = {0:[n0,0],1:[nL,h1],2:[nH,h2],3:[n3,hcavity[hh]]}
    z=0
    M = np.zeros((2, 2, len(wavelength)), dtype=np.complex128)
    Ma = np.zeros((2, 2, len(wavelength)), dtype=np.complex128)
    M[0,0,:] = 1
    M[1,1,:] = 1
    
    for jj in range(1,len(structure)):
        wi,h=param.get(structure[jj])
        ki = 2*np.pi*wi/wavelength
        wim,h=param.get(structure[jj-1])
        kim = 2*np.pi*wim/wavelength
        z=z+h
        Ma[0,0,:] = 0.5 * np.exp(-1j*(kim-ki)*z)*(1+wim/wi)
        Ma[0,1,:] = 0.5 * np.exp(1j*(kim+ki)*z)*(1-wim/wi)
        Ma[1,0,:] = 0.5 * np.exp(-1j*(kim+ki)*z)*(1-wim/wi)
        Ma[1,1,:] = 0.5 * np.exp(1j*(kim-ki)*z)*(1+wim/wi)
        for ii in range(0,len(wavelength)):
            M[:,:,ii]=Ma[:,:,ii].dot(M[:,:,ii])
    Emin = -1 * M[1,0,:]/M[1,1,:]

    Epout = (M[0,0,:]-M[0,1,:]*M[1,0,:]/M[1,1,:])
    Pmin.append(Emin*np.conj(Emin))
    Pout.append(Epout*np.conj(Epout)*wi/n0)


Pminc = np.reshape(Pmin,(len(hcavity),len(wavelength)))
Poutc = np.reshape(Pout,(len(hcavity),len(wavelength)))
mp.figure(1)
mp.pcolor(hc/wavelength,hcavity/hc,np.real(Pminc),shading='nearest')
mp.colorbar()
mp.xlabel("h/$\\lambda_0$")
mp.ylabel("$h_{cavity}$/h")

mp.figure(2)
mp.pcolor(hc/wavelength,hcavity/hc,10*np.log10(np.real(Pminc)),shading='nearest')
mp.colorbar()
mp.xlabel("h/$\\lambda_0$")
mp.ylabel("$h_{cavity}$/h")


mp.figure(4)
mp.pcolor(hc/wavelength,hcavity/hc,np.real(Poutc),shading='nearest')
mp.colorbar()
mp.xlabel("h/$\\lambda_0$")
mp.ylabel("$h_{cavity}$/h")

mp.show()

