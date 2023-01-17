"""
===============================================
Downloading topography data from OpenTopography
===============================================

Please find below a typical script allowing to download SRTM Global 3-arcsecond data from OpenTopography.
"""

from PyDune.data_processing.topography import downloadGlobalOpenTopo as GlobalOT

south,north,east,west = 21.25,21.35,54.45,54.55
bounds = (south,north,east,west)
loc = GlobalOT.getting_topography_data(bounds,demtype='SRTMGL3')