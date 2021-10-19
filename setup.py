from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="asct",
    version="0.0.3",
    author="Pete Canfield",
    author_email="pbczgf@umsystem.edu",
    description="Automated Single Cell Tuner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pbcanfield/ASCT",
    download_url='',
    license='MIT',
    install_requires=[
        'sbi',
        'neuron',
        'numpy',
        'scipy',
        'torch',
        'tqdm'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=['tests']),
    entry_points={
        'console_scripts': [
            'asct = asct.optimize_cell:main'
        ]
    }
)