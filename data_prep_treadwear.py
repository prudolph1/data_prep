'''
Author: Patrick Rudolph
Date: 12/10/19
Description: data error fixes and general cleaning to create modeling dataset
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = __import__('00_setup')

# import tread measurements
filepath = path.input + 'dataset.xlsx'
tread = pd.read_excel(filepath)

# records
print(tread.shape)

# last inspection date
print(tread['Month, Day, Year of Insp Date'].max())

# rename columns
tread = tread.rename(columns = {'TestId':'test_id',
                                'Insp':'inspection_num',
                                'VehicleNumber':'vehicle_num',
                                'Branding.1':'tire_id',
                                'Pattern':'pattern',
                                'RemovalReason':'removal_reason',
                                'Axle':'axle',
                                'TirePosition':'position',
                                'Month, Day, Year of Insp Date':'date',
                                'Mount Odometer':'mount_odometer',
                                'Avg. Odometer':'odometer',
                                'IP':'tire_pressure',
                                'RTD':'tread_depth',
                               })

# remove lost/missing tire records
tread = tread[tread['removal_reason'] != 'Lost or Missing']

# reformat type of inspectin_num to numeric
tread['inspection_num'] = tread['inspection_num'].replace('Mount','0').astype('int')

# reformat axle to front/rear
tread.loc[tread['axle'] == 1, 'axle'] = 'Front'
tread.loc[tread['axle'] == 2, 'axle'] = 'Rear'

# reformat position to left/right
tread.loc[tread['position'] == 1, 'position'] = 'Left'
tread.loc[tread['position'] == 2, 'position'] = 'Right'

# sort by location-vehicle-position-date
tread.sort_values(['test_id','vehicle_num','axle','position','date'], inplace = True)

# reset index
tread.reset_index(inplace = True, drop = True)

# force mount tire ID to match first inspection
tread['tire_id_next'] = tread['tire_id'].shift(-1)
cond1 = (tread['inspection_num'] == 0)
cond2 = (tread['tire_id'] != tread['tire_id_next'])
tread.loc[cond1 & cond2, 'tire_id'] = tread.loc[cond1 & cond2, 'tire_id_next']

# force mount tire odometer to match first inspection
tread['mount_odometer_next'] = tread['mount_odometer'].shift(-1)
cond1 = (tread['inspection_num'] == 0)
cond2 = (tread['mount_odometer'] != tread['mount_odometer_next'])
tread.loc[cond1 & cond2, 'odometer'] = tread.loc[cond1 & cond2, 'mount_odometer_next']

# set outlier pressures to missing
cond1 = (tread['tire_pressure'] > 150)
cond2 = (tread['tire_pressure'] == 0)
tread.loc[cond1 | cond2, 'tire_pressure'] = np.nan

# sort by location-vehicle-tire-date
tread.sort_values(['test_id','vehicle_num','tire_id','date'], inplace = True)

# import VINs
filepath = path.input + 'vehicle_vin_ref.xlsx'
vin = pd.read_excel(filepath)

# reset index
vin.reset_index(inplace = True)

# rename first column to location
vin.rename(inplace = True, columns = {'index':'location',
                                      'Unit':'vehicle_num',
                                      'License Plate #':'license_plate',
                                      'VIN':'vin'})

# drop missing rows and columns
vin.dropna(inplace = True, how = 'all', axis = 0)
vin.dropna(inplace = True, how = 'all', axis = 1)

# subset columns
vin = vin[['location','vehicle_num','vin']]

# merge vin onto tread
tread = tread.merge(vin, on = 'vehicle_num')

# reorder columns
tread = tread[['test_id','location','vehicle_num','make','model','vin','pattern','tire_id','axle','position','inspection_num','date','odometer','tread_depth','tire_pressure']]

# first record flag
cond1 = (tread['test_id'] != tread['test_id'].shift(1))
cond2 = (tread['vehicle_num'] != tread['vehicle_num'].shift(1))
cond3 = (tread['tire_id'] != tread['tire_id'].shift(1))

first_record = (cond1 | cond2 | cond3)

# last record flag
cond1 = (tread['test_id'] != tread['test_id'].shift(-1))
cond2 = (tread['vehicle_num'] != tread['vehicle_num'].shift(-1))
cond3 = (tread['tire_id'] != tread['tire_id'].shift(-1))

last_record = (cond1 | cond2 | cond3)

# set odometer to null for increasing (both records) or if = 0
cond1 = (tread['odometer'] < tread['odometer'].shift(1))
cond2 = (tread['odometer'] > tread['odometer'].shift(-1))
cond3 = (tread['odometer'] == 0)
tread.loc[(~first_record & cond1) | (~last_record & cond2) | cond3, 'odometer'] = np.nan

# create prev record fields
tread['date_prev'] = tread['date'].shift(1)
tread['tread_depth_prev'] = tread['tread_depth'].shift(1)
tread['odometer_prev'] = tread['odometer'].shift(1)
tread['tire_pressure_prev'] = tread['tire_pressure'].shift(1)

# differentials
tread['treadwear'] = tread['tread_depth_prev'].sub(tread['tread_depth'])
tread['miles'] = tread['odometer'].sub(tread['odometer_prev'])

# set negative treadwear to 0
tread['treadwear'] = tread['treadwear'].apply(lambda x: 0 if x < 0 else x)

# initialize miles and treadwear
tread.loc[first_record, 'miles'] = 0
tread.loc[first_record, 'treadwear'] = 0

# set first prev values to null
tread.loc[first_record, 'date_prev'] = np.nan
tread.loc[first_record, 'tread_depth_prev'] = np.nan
tread.loc[first_record, 'odometer_prev'] = np.nan
tread.loc[first_record, 'tire_pressure_prev'] = np.nan

# cumulative miles and wear
tread['cuml_miles'] = tread.groupby(['test_id','vehicle_num','tire_id'], as_index = False)['miles'].cumsum(skipna = False)
tread['cuml_treadwear'] = tread.groupby(['test_id','vehicle_num','tire_id'], as_index = False)['treadwear'].cumsum(skipna = False)

# flag rotations
cond4 = (tread['axle'] == tread['axle'].shift(1))
cond5 = (tread['position'] == tread['position'].shift(1))
tread['rotation'] = (~first_record & (~cond4 | ~cond5)).astype('int')

# output clean data file
filepath = path.output + 'tread_data.csv'
tread.to_csv(filepath, index = False)