"""
=======================================
Downloading data from the ERA* datasets
=======================================

Please find below a typical script allowing to download data from the ERA5 dataset.
"""

from pydune.data_processing import getting_CDSdata

month = [f"{i:02d}" for i in range(1, 2)]
day = [f"{i:02d}" for i in range(1, 20)]
time = [f"{i:02d}:00" for i in range(0, 24)]
year = [f"{i}" for i in range(2000, 2002)]
area = [-16.65, 11.9, -16.66, 11.91]

variable_dic = {"product_type": ["reanalysis"], # this is an important line
                "data_format": "netcdf",
                "variable": ["10m_u_component_of_wind", "2m_temperature"],
                "month": month,
                "day": day,
                "time": time,
                "year": year,
                "area": area,
                "download_format": "unarchived"}

a = getting_CDSdata("reanalysis-era5-single-levels",
                     variable_dic, "Angola_coast_v10",
                     Nsplit=2,
                     file="info.txt",
                     on_grid=False,
                     dt_check=2,
                     dt_print=2)
