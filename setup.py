from setuptools import find_packages, setup

setup(name="pydune",
      packages=find_packages(),
      python_requires=">=3",
      install_requires=[
          "numpy", "matplotlib", "cdsapi", "scipy", "datetime",
          "windrose", "xhistogram", "requests"],
      url="https://cgadal.github.io/pydune/",
      author="Cyril Gadal",
      license="Apache-2.0",
      version="0.2",
      zip_safe=False,
      )
