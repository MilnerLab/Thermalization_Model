import sys
from pathlib import Path
import importlib
import numpy as np

path_plot = Path(__file__).resolve().parent / "Plots"
sys.path.append(str(path_plot))

from scipy.interpolate import BSpline, PPoly

params = importlib.import_module("01_Parameters")
DEFAULTS = getattr(params, "DEFAULTS", {})

pi = np.pi
where = np.where
K_to_meV = 8.63e-2
meV_to_Hz = 241.8e9
Hz_to_meV = 1.0 / meV_to_Hz
m_to_A = 1e10

# Parameters
c = 238  # Speed of sound in m/s
Delta = 8.65 * 0.08617  # Roton gap in meV (1 K ≈ 0.08617 meV)
k0 = 1.92  # Roton minimum momentum in Å^-1
mu = 0.16  # Effective mass in units of helium-4 atom mass
Gamma = 0.1  # Width of the spectral line 

epsilon0=1.5/2 #meV
sigma=3 #A
T_K=2 #K
T=T_K*K_to_meV #meV
atomic_mass=0.24  #/meV.A^2
m_He=4.0026*atomic_mass 
m_w=12*atomic_mass
interaction_velocity = (epsilon0**2 /m_He/m_w)**(1/4)  # Interaction velocity in Å.meV

# Planck constant in eV·s
hbar = 6.582e-16  # eV·s

# Convert speed of sound to Å·meV
c_A_meV = c * 1e10 * hbar * 1e3  # Convert m/s to Å/s and then to meV

# Transition points
k_maxon = 0.5  # Transition from phonon to maxon
k_maxon_roton = 1.5  # Transition from maxon to roton


############################################################################################################
#Following Donnelly1981 

# --- Data Setup ---

# Control point coefficients A_i for i=1,...,10.
A = np.array([
    -2.4859411E-04,  # A1
     6.0320117E-01,  # A2
     3.6876093E+00,  # A3
     1.4997891E+01,  # A4
     1.4808346E+01,  # A5
     5.9384073E+00,  # A6
     1.6501400E+01,  # A7
     1.7724548E+01,  # A8
     1.8436561E+01,  # A9
     1.8435450E+01   # A10
])

# Knot vector:
# For a cubic B-spline (degree 3) with 10 control points we need 10+3+1 = 14 knots.
# First 4 knots are 0 (Q1, Q2, Q3, Q4) and then interior knots come from the table.
# For i=1,...,9 the table gives Q_{i+2}; for i=10 the knot is not provided.
# For a clamped spline, we set the last four knots equal.
t = np.array([
    0.0, 0.0, 0.0, 0.0,    # Q1, Q2, Q3, Q4
    0.0993,              # Q5 (from i=3)
    0.5100,              # Q6 (from i=4)
    1.6000,              # Q7 (from i=5)
    2.0230,              # Q8 (from i=6)
    2.4200,              # Q9 (from i=7)
    2.6650,              # Q10 (from i=8)
    3.6000,              # Q11 (from i=9)
    3.6000,              # Q12 (assumed, clamped)
    3.6000,              # Q13 (assumed, clamped)
    3.6000               # Q14 (assumed, clamped)
])

degree = 3  # Cubic B-spline

# Filename to store precomputed piecewise polynomial data.
filename = Path(str(DEFAULTS.get("spline_poly_path", "data/02_bath_and_angulon/spline_poly.npz")))
filename.parent.mkdir(parents=True, exist_ok=True)

# --- Load or Compute Piecewise Polynomial ---

if filename.exists():
    # Load precomputed polynomial coefficients and breakpoints.
    data = np.load(filename)
    c = data['c']
    x_breaks = data['x']
    piecewise_poly = PPoly(c, x_breaks)
    #print("Loaded precomputed piecewise polynomial from file.")
else:
    # Compute the B-spline and convert to a piecewise polynomial.
    spline_data = (t, A, degree)
    piecewise_poly = PPoly.from_spline(spline_data)
    # Save the coefficients and breakpoints for future use.
    np.savez(filename, c=piecewise_poly.c, x=piecewise_poly.x)
    print("Computed and saved the piecewise polynomial.")

# --- Evaluation and Plotting ---

# The valid parameter range is given by the breakpoints in piecewise_poly.x.
x_min = piecewise_poly.x[0]
x_max = piecewise_poly.x[-1]

# Dispersion relation
def omega_k_he(k):
    return where(k < x_min, 0, where(k>x_max, piecewise_poly(x_max), piecewise_poly(k) )) * K_to_meV

# Lorentzian profile function
def propagator_Lorentzian(k, omega, omega_k, Gamma=Gamma, m=atomic_mass):
    omega_k_vec = np.vectorize(omega_k)
    return   epsilon0 * k**2 /m / (omega**2 - omega_k_vec(k)**2 - 1j * omega * Gamma)

def propagator_He(k, omega): return propagator_Lorentzian(k, omega, np.vectorize(omega_k_he), Gamma=Gamma, m=m_He)
