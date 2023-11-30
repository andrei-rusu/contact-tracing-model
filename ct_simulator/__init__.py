# Define the metadata attributes
__version__ = '0.2.0'
__author__ = 'Andrei C. Rusu (andrei-rusu)'
__license__ = 'MIT'
__description__ = 'A Python package that can simulate the effects of various contact tracing strategies on the spread of viruses.'
__all__ = ['run_tracing', 'mcmc', 'tracing', 'Network']

from .tracing.network import Network