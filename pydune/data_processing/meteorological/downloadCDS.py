"""
This module allows to dowload data from the ERA5 and ERA5Land datasets hosted at
the `Climate Data Store <https://cds.climate.copernicus.eu/#!/home>`_ . Before
using this module, please read the corresponding documentation
`here <https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5>`_,
and especially the `part 4 <https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5#HowtodownloadERA5-4-DownloadERA5familydatathroughtheCDSAPI>`_.

Roughly, the steps are:

.. line-block::
    - create a CDS account `here <https://cds.climate.copernicus.eu/user/register>`_
    - install the CDS API, typically using `pip3 install cdsapi`
    - install the CDS API key corresponding to your account on your computer,
    following the steps described `here <https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS>`_

..warning:
    Still experimental, and scarce documentation.
"""

import datetime as dt
import decimal as dc
import os

import cdsapi
import numpy as np
import scipy.io as scio


def getting_CDSdata(dataset, variable_dic, name, Nsplit=1, file="info.txt", on_grid=True):
    """ This fuction helps to download data from datasets stored in the Climate Data Store.
         It splits data per integer year and then merge the data back together. This is sufficient at the moment
         as for most dataset, you can download a whole year of a single variable.

    Parameters
    ----------
    dataset : int
        dataset in which downloading the data.
        It can be 'reanalysis-era5-single-levels' or 'reanalysis-era5-land' for now.
    variable_dic : dic
        variable dictionnary to provide as a request.
    name : str
        name used to label the downloaded files.
    Nsplit : int
        number of requests in which the main request is split (the default is 1).
        If too small, will be corrected automatically.
    file : str
        filename under which some information about the request will be saved (the default is 'info.txt').
    on_grid : bool
        if True, the required coordinates will be matched with the native grid
        of the requested dataset. Otherwise, the dataset will be downloaded at
        the requested coordinates, using the interpolation of the CDS server (the default is True).

    Returns
    -------
    file_names : list
        the list of downloaded file names

    Examples
    --------
    >>> month = [i for i in range(1, 13)]
    >>> day = [i for i in range(1, 32)]
    >>> time = [i for i in range(0, 24)]
    >>> year = [i for i in range(1950, 2023)]
    >>> area = [-16.65, 11.9, -16.66, 11.91]
    >>> variable_dic = {'format': 'netcdf',
                    'variable': ['v10'],
                    'month': month,
                    'day': day,
                    'time': time,
                    'year': year,
                    'area': area,
                    'grid': [1.0, 1.0]}
    >>> a = CDS.Getting_wind_data('reanalysis-era5-land',
                              variable_dic, 'Angola_coast_v10',
                              Nsplit=6,
                              file='info.txt',
                              on_grid=False)

    """
    Names = {"reanalysis-era5-single-levels": "ERA5",
             "reanalysis-era5-land": "ERA5Land"}
    Nitems_max = {"reanalysis-era5-single-levels": 121000,
                  "reanalysis-era5-land": 12000,
                  "reanalysis-era5-pressure-levels": 60000}
    area_ref = [0, 0]
    #
    if Nsplit < 1:
        Nsplit = 1
    Nitems = len(variable_dic["variable"]) * (365.25 * len(variable_dic["month"])/12 * len(variable_dic["day"])/31) \
        * len(variable_dic["time"]) * len(variable_dic["year"])
    if Nitems/Nsplit > Nitems_max[dataset]:
        Nsplit = round(Nitems/Nitems_max[dataset]) + 1
        print("Request too large. Setting Nsplit =", Nsplit)

    # Puting the required area on the ERA5 grid
    area_wanted = variable_dic["area"]
    if on_grid:
        area_wanted[0] = area_wanted[0] - float(dc.Decimal(
            str(area_wanted[0] - area_ref[0])) % dc.Decimal(str(variable_dic["grid"])))
        area_wanted[1] = area_wanted[1] - float(dc.Decimal(
            str(area_wanted[1] - area_ref[1])) % dc.Decimal(str(variable_dic["grid"])))
        area_wanted[2] = area_wanted[2] - float(dc.Decimal(
            str(area_wanted[2] - area_ref[0])) % dc.Decimal(str(variable_dic["grid"])))
        area_wanted[3] = area_wanted[3] - float(dc.Decimal(
            str(area_wanted[3] - area_ref[1])) % dc.Decimal(str(variable_dic["grid"])))
        #
        variable_dic["area"] = area_wanted

    print("Area is :", area_wanted)
    #
    # Spliting request
    dates = np.array([int(i) for i in variable_dic["year"]])
    year_list = [list(map(str, j)) for j in np.array_split(dates, Nsplit)]
    #
    # checking the Nitems for every Nsplit
    Nitems_list = np.array([len(variable_dic["variable"]) * (365.25 * len(variable_dic["month"]) / 12
                                                             * len(variable_dic["day"]) / 31)*len(variable_dic["time"])
                                                             * len(i)
                            for i in year_list])
    if (Nitems_list > Nitems_max[dataset]).any():
        Nsplit = Nsplit + 1
        year_list = [list(map(str, j)) for j in np.array_split(dates, Nsplit)]
    #
    # filtering year_list
    year_list = [i for i in year_list if len(i) > 0]
    # Launching requests by year bins
    file_names = []
    for years in year_list:
        variable_dic["year"] = years
        if len(years) > 0:
            string = years[0] + "to" + years[-1]
        else:
            string = years[0]
        print(string)
        file_names.append(Names[dataset] + string +
                          "_" + name + "." + variable_dic["format"])
        client = cdsapi.Client()
        client.retrieve(dataset, variable_dic, file_names[-1])
    # Writing informations to spec file
    _save_spec_to_txt(dataset, variable_dic, file)
    return file_names


def load_netcdf(files_list):
    """ This function loads and concatenate (along the time axis) several NETCDF
    files from a list of filenames.

    Parameters
    ----------
    files_list : list
        the list of downloaded file names.

    Returns
    -------
    data : dict
        a dictionnary containing all data, concatenated along the time axis.

    """
    Data = {}
    for j, file in enumerate(files_list):
        file_temp = scio.netcdf.NetCDFFile(file, "r", maskandscale=True)
        for key in file_temp.variables.keys():
            if key not in Data.keys():
                Data[key] = file_temp.variables[key][:]
            elif key not in ["latitude", "longitude"]:
                Data[key] = np.concatenate(
                    (Data[key], file_temp.variables[key][:]), axis=0)
    #
    Data["time"] = _convert_time(Data["time"].astype(np.float64))
    # ## sort with respect to time
    sortedtime_inds = Data["time"].argsort()
    for key in Data.keys():
        if key not in ["latitude", "longitude"]:
            Data[key] = Data[key][sortedtime_inds, ...]
    return Data


def _save_spec_to_txt(dataset, variable_dic, file):
    if os.path.isfile(file):
        print(file + " already exists")
    else:
        with open(file, "w") as f:
            f.write("dataset: " + dataset + "\n")
            f.write("\n")
            for key in sorted(variable_dic.keys()):
                f.write(str(key) + ": " + str(variable_dic[key]) + "\n")
                f.write("\n")

# def Extract_points(points, file_format = 'npy', system_coordinates = 'cartesian'):
#     ######## function to extract specific points and write (u, v) velocity to <format> files
#     # points can either be a list of integers (1 is top left of the grid), or a list of coordinates (lat, lon)
#     # file_format is 'npy' or 'txt'
#     # system_coordinates is cartesian or polar
#     points = np.array(points)
#     if system_coordinates == 'polar':
#         self.Cartesian_to_polar()
#     for i, coords in points:
#         ## if for referencing point system
#         if ((len(points.shape) == 2) & (points.shape[-1] == 2)):
#             lat_ind = np.argwhere(coords[0] == self.latitude)[0][0]
#             lon_ind = np.argwhere(coords[1] == self.longitude)[0][0]
#             indexes = _sub2ind(self.Uwind[:-1], lat_ind, lon_ind)
#         else:
#             lat_ind, lon_ind = _ind2sub(self.Uwind[:-1], coords)
#             indexes = coords
#         #
#         # if for data coordinate system
#         if system_coordinates == 'cartesian':
#             data_to_write = [self.Uwind[lat_ind, lon_ind, :], self.Vwind[lat_ind, lon_ind, :]]
#         elif system_coordinates == 'polar':
#             data_to_write = [self.Ustrength[lat_ind, lon_ind, :], self.Uorientation[lat_ind, lon_ind, :]]
#         #
#         # if for saved file format
#         if file_format == 'npy':
#             np.save('Point_' + str(indexes) + '.npy', )
#         else:
#             np.savetxt('Point_' + str(indexes) + '.txt', [self.Uwind[lat_ind, lon_ind, :], self.Vwind[lat_ind, lon_ind, :]])

#############################################################################################


def _format_time(date):
    return f"{date[0]:04d}" + "-" + f"{date[1]:02d}" + "-" + f"{date[2]:02d}"


def _file_lenght(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def _convert_time(Times):
    atmos_epoch = dt.datetime(1900, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    # convert array of times in hours from epoch to dates
    return np.array([atmos_epoch + dt.timedelta(hours=i) for i in Times])


def _sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def _ind2sub(array_shape, ind):
    rows = (ind.astype("int") / array_shape[1])
    # or numpy.mod(ind.astype('int'), array_shape[1])
    cols = (ind.astype("int") % array_shape[1])
    return (rows, cols)
