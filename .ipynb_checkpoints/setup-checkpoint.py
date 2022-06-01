from setuptools import setup, find_packages

setup(name="PyDune",
      packages=find_packages(),
      python_requires='>=3',
      install_requires=[
        "numpy", "matplotlib", "cdsapi", "scipy", "datetime",
        "windrose", "xhistogram"],
      url='https://cgadal.github.io/PyDune/',
      author='Cyril Gadal',
      license='Apache-2.0',
      version='0.1',
      zip_safe=False,
      )
