import setuptools

setuptools.setup(
    name="lib_pprpa",
    version="0.1.0",
    url="https://github.com/lijiachen417/lib_pprpa",
    author="Jiachen Li",
    author_email="jiachen.li@yale.edu",
    description="Gives 2 electron addition/removal energies for a given molecule",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=["numpy", "h5py", "scipy", "pyscf"],
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7'
    ],
)
