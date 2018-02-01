from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vaipy',
    version='0.0.1',
    description='Personally Handcrafted Library for AI',
    long_description=long_description,
    url='https://github.com/svaisakh/vaipy',
    author='Vaisakh',
    author_email='svaisakh1994@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='ai artificial intelligence machine learning python library jupyter notebook',
    packages=find_packages(),
    install_requires=['pytest>=3.2.1', 'scipy>=0.19.1', 'matplotlib>=2.1.0', 'numpy>=1.13.3', 'hypothesis>=3.38.5'],
    python_requires='~=3.6',
)
