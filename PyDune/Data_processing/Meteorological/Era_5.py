# @Author: gadal
# @Date:   2019-05-21T18:44:14+02:00
# @Email:  gadal@ipgp.fr
# @Last modified by:   gadal
# @Last modified time: 2021-03-02T14:23:56+01:00

import cdsapi
import os
import numpy as np
from itertools import islice
from decimal import Decimal
from scipy.io import netcdf
from datetime import datetime, timezone, timedelta

"""
test
"""


def Getting_wind_data(dataset, variable_dic, name, Nsplit=1, file='info.txt', on_grid=True):
    Names = {'reanalysis-era5-single-levels': 'ERA5', 'reanalysis-era5-land': 'ERA5Land'}
    Nitems_max = {'reanalysis-era5-single-levels': 120000, 'reanalysis-era5-land': 100000}
    area_ref = [0, 0]
    #
    if Nsplit < 1:
        Nsplit = 1
    Nitems = len(variable_dic['variable']) * (365.25 * len(variable_dic['month'])/12 * len(variable_dic['day'])/31) \
        * len(variable_dic['time']) * len(variable_dic['year'])
    if Nitems/Nsplit > Nitems_max[dataset]:
        Nsplit = round(Nitems/Nitems_max[dataset]) + 1
        print('Request too large. Setting Nsplit =', Nsplit)

    # Defining years for data, either from dic variable
    dates = np.array([int(i) for i in variable_dic['year']])

    # Puting the required area on the ERA5 grid
    area_wanted = variable_dic['area']
    if on_grid:
        area_wanted[0] = area_wanted[0] - float(Decimal(str(area_wanted[0] - area_ref[0])) % Decimal(str(variable_dic['grid'])))
        area_wanted[1] = area_wanted[1] - float(Decimal(str(area_wanted[1] - area_ref[1])) % Decimal(str(variable_dic['grid'])))
        area_wanted[2] = area_wanted[2] - float(Decimal(str(area_wanted[2] - area_ref[0])) % Decimal(str(variable_dic['grid'])))
        area_wanted[3] = area_wanted[3] - float(Decimal(str(area_wanted[3] - area_ref[1])) % Decimal(str(variable_dic['grid'])))
        #
        variable_dic['area'] = area_wanted

    print('Area is :', area_wanted)
    #
    # Spliting request
    dates = np.array([int(i) for i in variable_dic['year']])
    year_list = [list(map(str, j)) for j in np.array_split(dates, Nsplit)]
    #
    # checking the Nitems for every Nsplit
    Nitems_list = np.array([len(variable_dic['variable']) * (365.25 * len(variable_dic['month'])/12 * len(variable_dic['day'])/31)*len(variable_dic['time']) * len(i)
                            for i in year_list])
    if (Nitems_list > Nitems_max).any():
        Nsplit = Nsplit + 1
        year_list = [list(map(str, j)) for j in np.array_split(dates, Nsplit)]
    #
    # Launching requests by year bins
    file_names = []
    for years in year_list:
        string = years[0] + 'to' + years[-1]
        print(string)
        file_names.append(Names[dataset] + string + '_' + name + '.' + variable_dic['format'])
        c = cdsapi.Client()
        variable_dic['year'] = years
        c.retrieve(dataset, variable_dic, file_names[-1])
    # Writing informations to spec file
    Save_spec_to_txt(dataset, variable_dic, file)
    return file_names


def Save_spec_to_txt(dataset, variable_dic, file):
    if os.path.isfile(file):
        print(file + ' already exists')
    else:
        with open(file, "w") as f:
            f.write('dataset: ' + dataset + '\n')
            f.write('\n')
            for key in sorted(variable_dic.keys()):
                f.write(str(key) + ': ' + str(variable_dic[key]) + '\n')
                f.write('\n')


def Load_netcdf(files_list):
    Data = {}
    for j, file in enumerate(files_list):
        file_temp = netcdf.NetCDFFile(file, 'r', maskandscale=True)
        for key in file_temp.variables.keys():
            if key not in Data.keys():
                Data[key] = file_temp.variables[key][:]
            elif key not in ['latitude', 'longitude']:
                Data[key] = np.concatenate((Data[key], file_temp.variables[key][:]), axis=0)
    #
    Data['time'] = Convert_time('time'.astype(np.float64))
    return Data



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
#             indexes = sub2ind(self.Uwind[:-1], lat_ind, lon_ind)
#         else:
#             lat_ind, lon_ind = ind2sub(self.Uwind[:-1], coords)
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

################################################################################
# Google earth functions
################################################################################


def Create_KMZ(name, coordinates):
    loc_path = os.path.join(os.path.dirname(__file__), 'src')
    # Destination file
    with open(name + '.kml', 'w') as dest:
        # Writing the first Part
        with open(os.path.join(loc_path, 'En_tete_era5.kml'), 'r') as entete:
            for line in islice(entete, 10, None):
                if line == '	<name>Skeleton_Coast.kmz</name>' + '\n':  # Premiere occurence
                    line = ' 	<name>' + name + '.kmz</name>' + '\n'
                elif line == '		<name>Skeleton_Coast</name>' + '\n':  # Second occurence
                    line = ' 	<name>' + name + '</name>'+'\n'
                dest.write(line)
        #
        # Writing placemarks
        with open(os.path.join(loc_path, 'placemark.kml'), 'r') as placemark:
            for i, Coord in enumerate(coordinates):
                lat, lon = Coord
                lon = Coord[1]
                print('lat =', lat)
                print('lon =', lon)
                #
                for line in islice(placemark, 7, None):
                    if line == '			<name>1</name>' + '\n':
                        line = '			<name>' + str(i + 1) + '</name>' + '\n'
                    if line == '				<coordinates>11.25,-17.25,0</coordinates>' + '\n':
                        line = '				<coordinates>' + lon + ',' + lat + ',0</coordinates>' + '\n'
                    dest.write(line)
                placemark.seek(0, 0)

        # Wrtiting closure
        with open(os.path.join(loc_path, 'bottom_page.kml'), 'r') as bottom:
            dest.writelines(bottom.readlines()[7:])

#############################################################################################


def format_time(date):
    return '{:04d}'.format(date[0]) + '-' + '{:02d}'.format(date[1]) + '-' + '{:02d}'.format(date[2])


def file_lenght(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def Convert_time(Times):
    atmos_epoch = datetime(1900, 1, 1, 0, 0, tzinfo=timezone.utc)
    # convert array of times in hours from epoch to dates
    return np.array([atmos_epoch + timedelta(hours=i) for i in Times])


def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)
