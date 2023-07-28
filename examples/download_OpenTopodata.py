"""
===============================================
Downloading topography data from OpenTopography
===============================================

Please find below a typical script allowing to download SRTM Global 3-arcsecond data from OpenTopography.
"""

from pydune.data_processing import getting_GOT_topography_data

south,north,east,west = 21.25,21.35,54.45,54.55
bounds = (south,north,east,west)
loc = getting_GOT_topography_data(bounds, demtype='SRTMGL3')