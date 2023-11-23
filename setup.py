#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ct_simulator',
    version='0.2.0',
    author='Andrei C. Rusu (andrei-rusu)',
    description='A Python package that can simulate the effects of various contact tracing strategies on the spread of viruses.',
    long_description=open('README.md','rt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/andrei-rusu/contact-tracing-model',
    project_urls={
        'Bug Tracker': 'https://github.com/andrei-rusu/contact-tracing-model/issues',
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    python_requires='>=3.9, <3.10',
    install_requires=[
        'matplotlib', 
        'numpy', 
        'networkx',
        'tqdm',
        'pandas',
    ],
    extras_require={
        'draw': [
            'ipython',
            'pygraphviz', # used for graphviz layouts (requires graphviz to be installed)
            'pyvis', # used for interactive visualizations with pyvis
            'plotly', # used for interactive visualizations with plotly
            'imageio', # used for saving animations
        ],
        'control': [
            'control_diffusion @ git+https://github.com/andrei-rusu/control-diffusion.git#egg=control_diffusion'
        ],
        'control_learn': [
            'control_diffusion @ git+https://github.com/andrei-rusu/control-diffusion.git#egg=cotnrol_diffusion[learn]'
        ],
        'mcmc': [
            'pymc3',
            'theano',
        ],
    },
    entry_points={
        'console_scripts': [
            'run-tracing=ct_simulator.run_tracing:main',
        ],
    },
)