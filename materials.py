from dataclasses import dataclass, field
import numpy as np
from typing import Callable

@dataclass
class MaterialProperties:
    """
    Material properties: E in MPa (N/mm²), dimensionless nu, α in 1/K, sigma_f in MPa
    """
    E: float        # Young's modulus [MPa]
    nu: float       # Poisson's ratio
    alpha: float    # Coefficient of thermal expansion [1/K]
    sigma_f: float  # Fracture/tensile strength [MPa]
    thermal_strain_4k: float = 0 # The thermal strain cooled down to 4K from 293 K

@dataclass
class LithiumNiobateProperties:
    """
    Anisotropic properties of Lithium Niobate
    """

    thermal_strain_4k_c_axis: float = -0.000633
    thermal_strain_4k_a_axis: float = -0.00244

    def get_cryo_stiffness_matrix():
        """
            Returns the Voigt stiffness matrix of Lithium Niobate at 6K in MPa
        """

        # Values come from Tarumi et al.
        _c11, _c12, _c13 = 205.6, 56.9, 69.9   # [GPa]
        _c14, _c33, _c44 = 8.0, 240, 62.0    # [GPa]
        _c66 = 74.3

        C_LINBO3_CRYO= np.array([
            [_c11,  _c12,  _c13,  _c14,  0.0,   0.0 ],
            [_c12,  _c11,  _c13, -_c14,  0.0,   0.0 ],
            [_c13,  _c13,  _c33,  0.0,   0.0,   0.0 ],
            [_c14, -_c14,  0.0,   _c44,  0.0,   0.0 ],
            [0.0,   0.0,   0.0,   0.0,   _c44,  _c14],
            [0.0,   0.0,   0.0,   0.0,   _c14,  _c66],
        ])

        # Convert to MPa
        return C_LINBO3_CRYO * 1000


class Materials:
    """
    Cryogenic values (4 K)
    ----------------------

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
    LITHIUM_NIOBATE_4K = LithiumNiobateProperties()
    ALUMINUM_4K        = MaterialProperties(E=78_000,  nu=0.33, alpha=0.5e-6, sigma_f=550, thermal_strain_4k=-415.45E-5 )
    MACOR_4K           = MaterialProperties(E=70_000,  nu=0.29, alpha=0.2e-6, sigma_f=80, thermal_strain_4k=-0.001692)   # ESTIMATE
    EPOXY_353ND_4K     = MaterialProperties(E=9_000,   nu=0.35, alpha=5e-6,   sigma_f=35, thermal_strain_4k=-0.014)   # VERY UNCERTAIN
    STAINLESS_STEEL_4K = MaterialProperties(E=210_000, nu=0.30, alpha=0.8e-6, sigma_f=1246) # NIST / Brookhaven (316L)


