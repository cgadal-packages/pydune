"""
===============================================
Downloading topography data from OpenTopography
===============================================

NOTE: This module has been moved as an example because the `gdal` python module can not be easily installed, which prevent from an appropriate maintenance of the `pydune` package.

This module allows to dowload data from global topography datasets hosted at
`OpenTopography <https://opentopography.org/>`_ . Before
using this module, please read the corresponding documentation
`here <https://portal.opentopography.org/apidocs/#/Public/getGlobalDem>`_,
and the data sources `here <https://portal.opentopography.org/dataCatalog?group=global>`_.

Roughly, the steps are:

    - create an OpenTopography account `here <https://portal.opentopography.org/newUser>`_
    - once logged in go to 'MyOpenTopo', then in the 'My Account' tab go to 'myOpenTopo Authorizations and API Key'
    - click 'Request API Key' and declare the string as your API_Key variable in Python

"""
import os
import requests
import numpy as np
from osgeo import gdal


def getting_GOT_topography_data(bounds, directory='./', demtype='AW3D30', API_Key='demoapikeyot2022'):
    """ This fuction downloads a GeoTiff from OpenTopography of a region of interest in a public global topography dataset.

    Parameters
    ----------
    bounds : tuple
        the bounding values for a rectangle where topography data. format is south, north, east, west.
    directory : str
        the directory where the GeoTiff should be saved (default value is the current directory).
    demtype : str
        the global dataset used. options are 'SRTMGL3','SRTMGL1','SRTMGL1_E','AW3D30','AW3D30_E','SRTM15Plus','NASADEM','COP30','COP90' (default value is 'AW3D30')
    API_Key : str
        the user API key for OpenTopography which can be found in the MyOpenTopo portal (default value is the demonstration key for 2022 which may expire 'demoapikeyot2022')

    Returns
    -------
    fname : str
        the downloaded file name including the full path

    Examples
    --------
    >>> south,north,east,west = 21.25,21.35,54.45,54.55
    >>> bounds = (south,north,east,west)
    >>> loc = GlobalOT.getting_topography_data(bounds,demtype='SRTMGL3')

    """
    south, north, east, west = bounds
    if east >= west:
        raise ValueError('east bound must be strictly less than west bound')
    if south >= north:
        raise ValueError('south bound must be strictly less than north bound')

    demtypes = ['SRTMGL3', 'SRTMGL1', 'SRTMGL1_E', 'AW3D30',
                'AW3D30_E', 'SRTM15Plus', 'NASADEM', 'COP30', 'COP90']
    if demtype not in demtypes:
        raise ValueError(
            'demtype must be one of the following:\n'+'\n'.join(demtypes))

    if not os.path.isdir(directory):
        raise ValueError('directory must exist')

    fname = 'OpenTopo_%s_S%s_N%s_E%s_W%s.tif' % (
        demtype, south, north, east, west)
    apiurl = 'https://portal.opentopography.org/API/globaldem?demtype=%s&south=%s&north=%s&west=%s&east=%s&outputFormat=GTiff&API_Key=%s' % (
        demtype, south, north, east, west, API_Key)
    r = requests.get(apiurl, stream=True)
    if r.status_code == 200:
        with open(directory+fname, 'wb') as f:
            f.write(r.content)
        print('Topography successfully downloaded.')
    elif r.status_code == 400:
        return 'Bad request to OpenTopography. Could be due to size; try decreasing bounds.'
    elif r.status_code == 401:
        return 'Unauthorized access to OpenTopography. Could be API Key.'
    elif r.status_code == 500:
        return 'Internal Error with OpenTopography.'
    return directory+fname


def load_xyz_geotiff(fname):
    """ This fuction loads a GeoTiff of topography data and returns coordinate arrays (x,y) and the elevation array (z).

    Parameters
    ----------
    fname : str
        the path to the GeoTiff file

    Returns
    -------
    x : numpy array
        the 1-d east-west coordinate array, is in the native units from the GeoTiff
    y : numpy array
        the 1-d south-north coordinate array, is in the native units from the GeoTiff
    z : numpy array
        the 2-d elevation array, origin is southeast corner, is in the native units from the GeoTiff
    """
    img = gdal.Open(fname)
    band = img.GetRasterBand(1)
    z = np.flipud(band.ReadAsArray())
    width = img.RasterXSize
    height = img.RasterYSize
    gt = img.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    x = np.linspace(minx, maxx, width)
    y = np.linspace(miny, maxy, height)
    return x, y, z


# %%
# Downloading topography data from OpenTopography
# ===============================================

# Please find below a typical script allowing to download SRTM Global 3-arcsecond data from OpenTopography.

if __name__ == '__main__':

    south, north, east, west = 21.25, 21.35, 54.45, 54.55
    bounds = (south, north, east, west)
    loc = getting_GOT_topography_data(bounds, demtype='SRTMGL3')
