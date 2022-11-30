import numpy as np
from matplotlib import pyplot as plt

# CONSTANTS
AVOGADRO = 6.022e23  # mol^(-1)
c = 3e+8  # m/s
E_RADIUS = 2.817e-13  # cm
E_MASS = 0.511 / c**2  # MeV/c^(2)

# FUNCTIONS
def fGamma(beta):
    return 1 / np.sqrt(1 - beta**2)

# CLASSES
class Element:
    def __init__(self, name, mass, charge, density, I):
        self._name = name
        self._A = mass
        self._Z = charge
        self._density = density
        self._I = I
        self._momentum = 0

    def __str__(self):
        info = (
            f'\nParticle name: {self._name}\n'
            f'Atomic mass = {self._A}\n'
            f'Atomic number (Z) = {self._Z}\n'
            f'Density = {self._density:.2f} g/cm^3\n'
            f'Ionization energy = {self._I} MeV'
        )
        if (self._momentum != 0):
            info += f'\nMomentum = {self._momentum} MeV/c'
        return info

    @property
    def A(self):
        return float(self._A) 

    @property
    def Z(self):
        return float(self._Z)

    @property
    def density(self):
        return float(self._density)

    @property
    def I(self):
        return float(self._I )

class Particle:
    def __init__(self, name, mass, charge):
        self._name = name
        self._mass = mass  # MeV / c^2
        self._charge = charge  # units of e
        self._momentum = float(0)  # MeV / c
    
    def __str__(self):
        info = (
            f'\nParticle name: {self._name}\n'
            f'Mass = {self._mass} MeV/c^2\n'
            f'Charge = {self._charge}\n'
            f'Momentum = {self._momentum:.2f} MeV/c\n'
            f'Energy = {self.energy:.2f} MeV'
        )
        return info

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, p):
        p = float(p)
        if (p < 0):
            raise TypeError()
        self._momentum = p

    @property
    def beta(self):
        energy = np.sqrt(self._momentum**2 + self._mass**2)
        return self._momentum / energy

    @beta.setter
    def beta(self, b):
        gamma = fGamma(b)
        self._momentum = self._mass * b * gamma

    @property
    def energy(self):
        return np.sqrt(self._momentum**2 + self._mass**2)

    @energy.setter
    def energy(self, E):
        self._momentum = np.sqrt(E**2 - self._mass**2)
    
    def Bethe(self, el):
        if (type(el) != Element):
            raise TypeError()
        beta = self.beta
        z = self._charge
        M = self._mass

        Z = el.Z
        A = el.A
        I = el.I
        rho = el.density

        k = 2 * np.pi * AVOGADRO * E_RADIUS**2 * E_MASS * c**2        
        gamma = fGamma(beta)
        v = beta * c
        s = E_MASS / M
        eta = beta * gamma
        W_max = (2 * E_MASS * c**2 * eta**2) / \
                (1 + 2 * s * np.sqrt(1 + eta**2) + s**2)
        
        dEdx = k * (Z / A) * (z / beta)**2 *\
               (np.log((2 * E_MASS * gamma**2 * v**2 * W_max) / I**2) - 2 * beta**2)
        
        return dEdx

if __name__ == "__main__":
    copper = Element("Cu", 63.5, 29, 8.96, 322e-6)
    proton = Particle("p", 938, 1)
    proton.momentum = 600

    print(proton.Bethe(copper))