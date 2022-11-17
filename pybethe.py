import numpy as np
from matplotlib import pyplot as plt

AVOGADRO = 6.022e23 # mol^(-1)
c = 3e+8 # m/s
E_RADIUS = 2.817e-13 # cm
E_MASS = 0.511 / c**2 # MeV/c^(2)

def fGamma(beta):
    return 1 / np.sqrt(1 - beta**2)

def fBetaP(pulse, mass):
    return pulse / np.sqrt( mass**2 + pulse**2)

def fBetaE(energy, mass):
    return np.sqrt(energy**2 - mass**2) / energy

def Bethe(Z, A, I, rho, z, beta, M): # (1 / rho) * dE/dx
    k = 2 * np.pi * AVOGADRO * E_RADIUS**2 * E_MASS * c**2

    gamma = 1 / np.sqrt(1 - beta**2)
    v = beta * c
    s = E_MASS / M
    eta = beta * gamma
    W_max = (2 * E_MASS * c**2 * eta**2) / (1 + 2 * s * np.sqrt(1 + eta**2) + s**2)

    stopping_power = k * (Z / A) * (z / beta)**2 *\
                     (np.log( (2 * E_MASS * gamma**2 * v**2 * W_max) / I**2) - 2 * beta**2)
    
    return stopping_power

if __name__ == "__main__":
    # Copper
    I_Cu = 322e-6 # MeV
    Z_Cu = 29
    A_Cu = 63.5
    rho_Cu = 8.96 # g/cm^(3)

    beta_p = fBetaP(410, 938)

    dEdx = Bethe(Z_Cu, A_Cu, I_Cu, rho_Cu, 1, beta_p, 938)
    print(dEdx)

    x = np.logspace(1, 6, 100, base=10)
    beta_p = fBetaP(x, 938)
    y = Bethe(Z_Cu, A_Cu, I_Cu, rho_Cu, 1, beta_p, 938)

    plt.errorbar(beta_p * fGamma(beta_p), y, fmt = '.r')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()