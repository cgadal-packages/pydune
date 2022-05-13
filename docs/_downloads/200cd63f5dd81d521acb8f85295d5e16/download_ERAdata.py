"""
=======================================
Downloading data from the ERA* datasets
=======================================

Please find below a typical script allowing to download data from the ERA5 Land dataset.
"""

from PyDune.Data_processing.Meteorological import downloadCDS as CDS

month = [i for i in range(1, 13)]
day = [i for i in range(1, 32)]
time = [i for i in range(0, 24)]
year = [i for i in range(1950, 2023)]
area = [-16.65, 11.9, -16.66, 11.91]

variable_dic = {'format': 'netcdf',
                'variable': ['v10'],
                'month': month,
                'day': day,
                'time': time,
                'year': year,
                'area': area,
                'grid': [1.0, 1.0]}

a = CDS.Getting_wind_data('reanalysis-era5-land',
                          variable_dic, 'Angola_coast_v10',
                          Nsplit=6,
                          file='info.txt',
                          on_grid=False)
