"""
StabPy - Stable Distributions and Long Memory Processes
========================================================

A Python package for working with stable distributions, Normal Inverse Gaussian
distribution, and FARIMA processes.

Modules
-------
stable : Stable distribution tools
normal : Normal distribution tests
nig : Normal Inverse Gaussian distribution
farima : FARIMA and fractional processes
"""

__version__ = "0.1.3"

# Import from stable module
from .stable import (
    stabcdf,
    stabcull,
    stabreg,
    stabrnd,
    stabtest
)

# Import from normal module
from .normal import normtest

# Import from NIG module
from .nig import (
    nigpdf,
    nigcdf,
    nigest,
    nigrnd,
    nigtest,
    invgrnd,
    nigloglik
)

# Import from FARIMA module
from .farima import (
    fracdiff,
    gam,
    usg,
    fftlfsn,
    fftFarimaEst,
    fftfarima,
    sstabrnd,
    IntegralEst
)

__all__ = [
    # Stable
    'stabcdf', 'stabcull', 'stabreg', 'stabrnd', 'stabtest',
    # Normal
    'normtest',
    # NIG
    'nigpdf', 'nigcdf', 'nigest', 'nigrnd', 'nigtest', 'invgrnd', 'nigloglik',
    # FARIMA
    'fracdiff', 'gam', 'usg', 'fftlfsn', 'fftFarimaEst', 'fftfarima', 'sstabrnd', 'IntegralEst',
]