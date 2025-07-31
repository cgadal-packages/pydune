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
import time

import cdsapi
import numpy as np
from netCDF4 import Dataset

SHORT_NAMES = {
    "reanalysis-era5-single-levels": "ERA5",
    "reanalysis-era5-pressure-levels": "ERA5_PLEVELS",
    "reanalysis-era5-land": "ERA5Land",
}
NITEMS_MAX = {
    "reanalysis-era5-single-levels": 121000,
    "reanalysis-era5-land": 12000,
    "reanalysis-era5-pressure-levels": 60000,
}
AREA_REF = [0, 0]


def _compute_item_number(variable_dic):
    return (
        len(variable_dic["variable"])
        * (365.25 * len(variable_dic["month"]) / 12 * len(variable_dic["day"]) / 31)
        * len(variable_dic["time"])
        * len(variable_dic["year"])
    )


def _compute_area_ongrid(variable_dic):
    area_wanted = [
        variable_dic["area"][i]
        - float(dc.Decimal(str(variable_dic["area"][i] - AREA_REF[i % 2])) % dc.Decimal(str(variable_dic["grid"])))
        for i in range(4)
    ]
    return area_wanted


def _launch_requests_onebyone(variable_dic, year_list, dataset, name):
    client = cdsapi.Client()
    file_names = []
    for years in year_list:
        variable_dic["year"] = years
        key = f"{years[0]}to{years[-1]}" if len(years) > 1 else years[0]
        print(key)
        filename = f"{SHORT_NAMES[dataset]}{key}_{name}.{variable_dic['data_format']}"
        client.retrieve(dataset, variable_dic).download(filename)
        #
        file_names.append(filename)
    return file_names


def build_rqst_status(requests):
    return {key: rqst.reply["state"] for key, rqst in requests.items()}


def check_rqst(requests):
    states = build_rqst_status(requests)
    status_array = np.array(list(states.values()))
    finished = (status_array == "completed") | (status_array == "failed")
    return not np.all(finished)


def print_request_status(requests):
    line = f"---- Request Status {time.ctime()} ----"
    print(line)
    for key, rqst in requests.items():
        state = rqst.reply["state"]
        print(f"    {key}: {state}")
        if state == "failed":
            print("     Message: %s", rqst.reply["error"].get("message"))
            print("     Reason:  %s", rqst.reply["error"].get("reason"))
    print("".join(["-" for i in range(len(line))]) + "\n")


def _launch_allrequests_directly(variable_dic, year_list, dataset, name, dt_check=30, dt_print=120):
    """
    Launches multiple CDS API requests and downloads results once all are completed.

    Parameters:
        variable_dic (dict): CDS API variable dictionary.
        year_list (list): List of lists of years (e.g., [["2020"], ["2021", "2022"]]).
        dataset (str): Dataset name for the CDS API.
        name (str): Custom name for output files.
        dt_check (int): Interval (in seconds) to check request status.
        dt_print (int): Interval (in seconds) to print status updates.
    """

    client = cdsapi.Client(wait_until_complete=False)
    requests = {}

    # ### launch all requests directly on the sever without waiting for them to complete
    for years in year_list:
        variable_dic["year"] = years
        string = f"{years[0]}to{years[-1]}" if len(years) > 1 else years[0]
        print(string)
        requests[string] = client.retrieve(dataset, variable_dic)

    # ### check periodically if the request is finished or not
    last_print_time = time.time()
    while check_rqst(requests):
        time.sleep(dt_check)
        for key, rqst in requests.items():
            rqst.update()

        current_time = time.time()
        if dt_print and current_time - last_print_time >= dt_print:
            print_request_status(requests)
            last_print_time = current_time
    print("All requests completed.")
    # ### once everything is finished, download all files
    file_names = []
    for key, rqst in requests.items():
        filename = f"{SHORT_NAMES[dataset]}{key}_{name}.{variable_dic['data_format']}"
        print(f"Downloading: {filename}")
        rqst.download(filename)
        file_names.append(filename)
    return file_names


def getting_CDSdata(
    dataset,
    variable_dic,
    name,
    Nsplit=1,
    file="info.txt",
    on_grid=False,
    all_requests_directly=True,
    dt_check=30,
    dt_print=120,
):
    """This fuction helps to download data from datasets stored in the Climate Data Store.
         It splits data per integer year and then merge the data back together. This is sufficient at the moment
         as for most dataset you can download a whole year of a single variable.

    Parameters
    ----------
    dataset : int
        dataset in which downloading the data.
        It can be 'reanalysis-era5-single-levels', 'reanalysis-era5-land' or "reanalysis-era5-pressure-levels" for now.
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
        the requested coordinates, using the interpolation of the CDS server (the default is False).
    all_requests_directly : bool
        if True, all requests are sent at once, and then status is checked periodically every `dt_check` until completion,
        and then all files are downloaded. If False, requests are processed and downloaded one by one (the default is True).
    dt_check : float
        seconds between every status check when `all_requests_directly` is True. (the default is 30)
    dt_print : float
        seconds between every status print when `all_requests_directly` is True. (the default is 120)

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
    #
    Nsplit = max(Nsplit, 1)
    Nitems = _compute_item_number(variable_dic)
    if Nitems / Nsplit > NITEMS_MAX[dataset]:
        Nsplit = round(Nitems / NITEMS_MAX[dataset]) + 1
        print("Request too large. Setting Nsplit =", Nsplit)
    #
    # Puting the required area on the ERA5 grid
    if on_grid:
        variable_dic["area"] = _compute_area_ongrid(variable_dic)
        print("Area has been modified to be on native grid.")
    print("Area is :", variable_dic["area"])
    #
    # Spliting request
    dates = np.array([int(i) for i in variable_dic["year"]])
    year_list = [list(map(str, j)) for j in np.array_split(dates, Nsplit)]
    #
    # checking the Nitems for every Nsplit
    Nitems_list = np.array(
        [
            len(variable_dic["variable"])
            * (365.25 * len(variable_dic["month"]) / 12 * len(variable_dic["day"]) / 31)
            * len(variable_dic["time"])
            * len(i)
            for i in year_list
        ]
    )
    if (Nitems_list > NITEMS_MAX[dataset]).any():
        Nsplit = Nsplit + 1
        year_list = [list(map(str, j)) for j in np.array_split(dates, Nsplit)]
    #
    # filtering year_list
    year_list = [i for i in year_list if len(i) > 0]
    print(f"The list of year chunks is: {year_list}")
    # Launching requests by year bins
    if (not all_requests_directly) | len(year_list) == 1:
        file_names = _launch_requests_onebyone(variable_dic, year_list, dataset, name)
    else:
        file_names = _launch_allrequests_directly(variable_dic, year_list, dataset, name,
                                                  dt_check=dt_check, dt_print=dt_print)
    # Writing informations to spec file
    _save_spec_to_txt(dataset, variable_dic, file)
    return file_names


def load_netcdf(files_list):
    """
    Load and concatenate (along the time axis) several NetCDF files.

    Parameters
    ----------
    files_list : list of str
        List of NetCDF file paths to load.

    Returns
    -------
    data : dict
        Dictionary containing all variables, concatenated along the time axis.
    """
    Data = {}
    for j, file in enumerate(files_list):
        with Dataset(file, mode="r") as ds:
            for key in ds.variables:
                var_data = ds.variables[key][:]
                if key not in Data:
                    Data[key] = var_data
                elif key not in ["latitude", "longitude"]:
                    Data[key] = np.concatenate((Data[key], var_data), axis=0)

    # Convert and sort time
    Data["time"] = _convert_time(Data["time"].astype(np.float64))
    sorted_inds = Data["time"].argsort()

    for key in Data:
        if key not in ["latitude", "longitude"]:
            Data[key] = Data[key][sorted_inds, ...]

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
    return rows * array_shape[1] + cols


def _ind2sub(array_shape, ind):
    rows = ind.astype("int") / array_shape[1]
    # or numpy.mod(ind.astype('int'), array_shape[1])
    cols = ind.astype("int") % array_shape[1]
    return (rows, cols)
