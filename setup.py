from impedance_extend import __version__
import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="impedance_extend",
    version=__version__,
    author="impedance_extend.py developers",
    author_email="prof.krishna.v@gmail.com",
    description="A package for analyzing electrochemical impedance data "
                "(with support for GA, PSO and least_squares or combinations "
                "thereof)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://impedance-extend.readthedocs.io/en/latest/",
    packages=setuptools.find_packages(),
    python_requires="~=3.10",
    install_requires=['altair>=3.0', 'matplotlib>=3.5',
                      'numpy>=1.22.4', 'scipy>=1.0',
                      'pandas', 'pygad>=3.6.0', 'pyswarms>=1.3'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
