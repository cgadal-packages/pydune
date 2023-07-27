"""
=======================================
Downloading data from the ERA* datasets
=======================================

Please find below a typical script allowing to download data from the ERA5 Land dataset.
"""

from PyDune.data_processing.meteorological import downloadCDS as CDS

month = [i for i in range(1, 2)]
day = [i for i in range(1, 20)]
time = [i for i in range(0, 24)]
year = [i for i in range(2000, 2001)]
area = [-16.65, 11.9, -16.66, 11.91]

variable_dic = {'format': 'netcdf',
                'variable': ['10m_u_component_of_wind', '2m_temperature'],
                'month': month,
                'day': day,
                'time': time,
                'year': year,
                'area': area}

a = CDS.getting_data('reanalysis-era5-single-levels',
                     variable_dic, 'Angola_coast_v10',
                     Nsplit=1,
                     file='info.txt',
                     on_grid=False)
