import numpy as np

# -------------------------
# Utility functions
# -------------------------

def cos_angle(a, b, min=-1, max=1):
    return np.clip(np.dot(a, b), min, max)

def safe_tan(x):
    return np.tan(np.clip(x, -np.pi/2+1e-6, np.pi/2-1e-6))

def safe_arccos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

# -------------------------
# Henyey-Greenstein phase function
# -------------------------
def HG_double_lobe(cosg, b, c):
    term1 = (1 + c)/2 * (1 - b*b) / (1 - 2*b*cosg + b*b)**1.5
    term2 = (1 - c)/2 * (1 - b*b) / (1 + 2*b*cosg + b*b)**1.5
    return term1 + term2

# -------------------------
# SHOE
# -------------------------
def B_S(tan_g_div_2, h_S):
    return 1.0 / (1.0 + tan_g_div_2 / h_S)

# -------------------------
# CBOE
# -------------------------
def B_C(tan_g_div_2, h_C):
    T = tan_g_div_2 / h_C
    if T < 1e-8:
        return 1.0
    return (1 + (1 - np.exp(-T))/T) / (2 * (1 + T)**2)

# -------------------------
# H-function (Ambartsumian–Chandrasekhar)
# -------------------------
def Hfunc(x, w):
    r0 = (1 - np.sqrt(1-w)) / (1 + np.sqrt(1-w))
    term = r0 + (1 - 2*r0*x)/2 * np.log(1+1/x)
    return 1.0 / (1 - w*x * term)

# -------------------------
# Porosity factor K
# -------------------------
def porosity_K(phi):
    return -np.log(1 - 1.209 * phi**0.667) / (1.209 * phi**0.667)

# -------------------------
# Surface roughness helper functions
# -------------------------
def chi(thetap):
    return 1.0 / np.sqrt(1 + np.pi * np.tan(thetap)**2)

def E1(thetap, cosy, siny):
    return np.exp(-2/np.pi * 1/np.tan(thetap)*cosy/siny)

def E2(thetap, cosy, siny):
    return np.exp(-1/np.pi * (1/np.tan(thetap)*cosy/siny)**2)

def eta(thetap, cosy, siny):
    return chi(thetap) * (cosy + siny*np.tan(thetap) * E2(thetap,cosy,siny)/(2 - E1(thetap,cosy,siny)))

def f_psi(psi):
    return np.exp(-2*np.tan(psi/2))
# -------------------------
# Core roughness geometry term S(i,e,ψ)
# -------------------------
def rough_S(cosi, cose, sini, sine, psi, thetap):
    chi_t = chi(thetap)
    tan_thetap = np.tan(thetap)
    sin_psi_half_square = np.sin(psi/2) ** 2

    if cosi >= cose:
        mu0 = chi_t * (cosi + sini*tan_thetap * (np.cos(psi)*E2(thetap,cose,sine) + sin_psi_half_square * E2(thetap,cosi,sini)) / (2 - E1(thetap,cose,sine) - (psi/np.pi)*E1(thetap,cosi,sini)))
        mue = chi_t * (cose + sine*tan_thetap * (E2(thetap,cose,sine) - sin_psi_half_square * E2(thetap,cosi,sini)) / (2 - E1(thetap,cose,sine) - (psi/np.pi)*E1(thetap,cosi,sini)))
        S = mue/eta(thetap, cose, sine) * cosi/eta(thetap, cosi, sini) * chi_t / (1 - f_psi(psi) + f_psi(psi)*chi_t*(cosi/eta(thetap, cosi, sini)))
    else:
        mu0 = chi_t * (cosi + sini*tan_thetap * (E2(thetap,cosi,sini) - sin_psi_half_square * E2(thetap,cose,sine)) / (2 - E1(thetap,cosi,sini) - (psi/np.pi)*E1(thetap,cose,sine)))
        mue = chi_t * (cose + sine*tan_thetap * (np.cos(psi)*E2(thetap,cosi,sini) + sin_psi_half_square * E2(thetap,cose,sine)) / (2 - E1(thetap,cosi,sini) - (psi/np.pi)*E1(thetap,cose,sine)))
        S = mue/eta(thetap, cose, sine) * cosi/eta(thetap, cosi, sini) * chi_t / (1 - f_psi(psi) + f_psi(psi)*chi_t*(cose/eta(thetap, cose, sine)))

    return mu0, mue, S

# -------------------------
# Full Hapke BRDF
# -------------------------
def hapke_BRDF(inc_angle, emi_angle, phase_angle, *, w, b, c, hS, B_S0, hC, B_C0, thetap, phi, **kwargs):
    """
    normal: surface normal (3,)
    light_dir: incident light direction (towards surface)
    ray_dir: view direction (towards camera)
    """

    # cosi = np.abs(cos_angle(normal, light_dir, -1e-6, 1))
    cosi = np.cos(np.deg2rad(inc_angle))
    cose = np.cos(np.deg2rad(emi_angle))
    sini = np.sin(np.deg2rad(inc_angle))
    sine = np.sin(np.deg2rad(emi_angle))
    cosg = np.cos(np.deg2rad(phase_angle))
    tan_g_div_2 = safe_tan(safe_arccos(cosg) / 2)

    psi = safe_arccos((cosg - cosi*cose) / (sini*sine + 1e-6))

    p_g = HG_double_lobe(cosg, b, c)

    BS = B_S(tan_g_div_2, hS)
    BC = B_C(tan_g_div_2, hC)

    K = porosity_K(phi)
    mu0, mue, S = rough_S(cosi, cose, sini, sine, psi, thetap)

    M = Hfunc(mu0/K, w) * Hfunc(mue/K, w) - 1
    LS = mu0 / (mu0 + mue)

    return (w/(4*np.pi)) * LS/cosi * K * (p_g*(1 + BS * B_S0) + M)*(1 + BC * B_C0) * S