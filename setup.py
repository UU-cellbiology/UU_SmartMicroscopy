'''
Example taken from user kmario23 in https://stackoverflow.com/questions/1471994/what-is-setup-py
'''
from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='UU_SmartMicroscopy',
   version='1.0',
   packages=find_packages(),
#    packages=['UU_SmartMicroscopy'],
   description='Modular python library to cconduct smart microscopy experiments',
   license="MIT",
   long_description=long_description,
   author='Alfredo Rates, et. al.',
   author_email='a.ratessoriano@uu.nl',
   url="https://github.com/UU-cellbiology/UU_SmartMicroscopy",
   install_requires=['numpy', 
                    'Pillow', 
                    'scipy',
                    'tifffile',
                    'threaded',
                    'datetime',
                    'tk',
                    'scikit-image',
                    'pathlib',
                    'importlib',
                    'PyYAML',
                    'pycromanager',
                    'matplotlib',
                    'torch',
                    'torchvision',
                    'segment_anything',
                    'urllib3',
                     ],
   scripts=[
            'useCases/Minimal/Minimal.py',
            'useCases/OutcomeDriven/OutcomeDriven.py',
           ]
)
