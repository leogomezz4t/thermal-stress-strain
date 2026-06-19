from dataclasses import dataclass

@dataclass
class MaterialProperties:
    """
    Material properties: E in MPa (N/mm²), dimensionless nu, α in 1/K, sigma_f in MPa
    """
    E: float        # Young's modulus [MPa]
    nu: float       # Poisson's ratio
    alpha: float    # Coefficient of thermal expansion [1/K]
    sigma_f: float  # Fracture/tensile strength [MPa]

class Materials:
    """
    Room-temperature values (293 K)
    --------------------------------
    DOUBLE CHECK — values below were AI-estimated; verify against datasheets / Ekin before trusting.
    alpha: coefficient of thermal expansion at ~293 K.
    sigma_f: tensile fracture strength for brittle materials; UTS for ductile ones.
    LiNbO3 is strongly anisotropic — isotropic E and alpha here are rough averages only.
    """
    LITHIUM_NIOBATE = MaterialProperties(E=170_000, nu=0.25, alpha=4e-6,   sigma_f=150)  # WRONG: orientation-dependent
    ALUMINUM        = MaterialProperties(E=70_000,  nu=0.33, alpha=23e-6,  sigma_f=270)  # alloy-dependent
    MACOR           = MaterialProperties(E=66_900,  nu=0.29, alpha=9.3e-6, sigma_f=94)
    EPOXY_353ND     = MaterialProperties(E=3_500,   nu=0.35, alpha=54e-6,  sigma_f=69)
    STAINLESS_STEEL = MaterialProperties(E=193_000, nu=0.29, alpha=17.3e-6, sigma_f=515)

    """
    Cryogenic values (4 K)
    ----------------------
    Use these for the stiffness matrix when solving at the cryogenic operating point.
    alpha here is the *instantaneous* value at 4 K (near zero for all materials).
    NOTE: do NOT use alpha_4K * ΔT as the thermal eigenstrain — that still underestimates
    the contraction.  The correct thermal strain is ∫α(T)dT from 4K→293K (sub-problem B).

    Confidence by material:
      ALUMINUM_4K        — medium-high.  E from NIST TRC 6061 curve (~13% increase).
                           sigma_f estimated; cryogenic UTS for 6061-T6 roughly doubles.
      STAINLESS_STEEL_4K — high for E and nu (NIST / Brookhaven data);
                           sigma_f is grade-dependent (316LN >> 316L).
      LITHIUM_NIOBATE_4K — LOW.  No public 4 K data found.  ~3% stiffening assumed
                           by analogy with other oxide ceramics.  MUST verify.
      MACOR_4K           — LOW.  No public 4 K data.  ~5% stiffening assumed.
                           MUST verify.
      EPOXY_353ND_4K     — VERY LOW.  Epoxies vary enormously between formulations.
                           E typically increases 2–4× at 4 K; sigma_f drops as epoxy
                           becomes brittle.  Contact Epo-Tek for data or measure directly.
    """
    LITHIUM_NIOBATE_4K = MaterialProperties(E=175_000, nu=0.25, alpha=0.5e-6, sigma_f=150)  # ESTIMATE
    ALUMINUM_4K        = MaterialProperties(E=79_000,  nu=0.33, alpha=0.5e-6, sigma_f=550)  # NIST TRC 6061
    MACOR_4K           = MaterialProperties(E=70_000,  nu=0.29, alpha=0.2e-6, sigma_f=80)   # ESTIMATE
    EPOXY_353ND_4K     = MaterialProperties(E=9_000,   nu=0.35, alpha=5e-6,   sigma_f=35)   # VERY UNCERTAIN
    STAINLESS_STEEL_4K = MaterialProperties(E=210_000, nu=0.30, alpha=0.8e-6, sigma_f=1246) # NIST / Brookhaven (316L)


