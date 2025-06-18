

import zipfile as zf
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from glob import glob
from scipy.interpolate import griddata
from tqdm import tqdm



# Your data folder with zipped SAFE files
data_folder = '/Users/silleprydshansen/Desktop/ /EO4C/EO4C/Project/downloads2'
zip_files = glob(os.path.join(data_folder, '*.SAFE.zip'))

# Lists to collect lat, lon, and wind data
lat_list, lon_list = [], []
speed_list, dir_list = [], []
datetime_list = []

for file_name in zip_files:
    with zf.ZipFile(file_name, 'r') as zipp:
        zipp.extractall('./temp_extracted')  # Extract to temp folder
        # Determine root SAFE folder
        safe_folders = set([f.split('/')[0] for f in zipp.namelist() if '.SAFE/' in f])
        if not safe_folders:
            raise ValueError(f"No .SAFE folder found in {file_name}")
        extracted_folder = list(safe_folders)[0]

        # Path to measurement data
        nc_path = os.path.join('./temp_extracted', extracted_folder, 'measurement')

        
        # Find the NetCDF file
        nc_file = [f for f in os.listdir(nc_path) if f.endswith('.nc')][0]
        dataset = nc.Dataset(os.path.join(nc_path, nc_file))
        
        # Load data
        owiLat = np.array(dataset.variables['owiLat'])
        owiLon = np.array(dataset.variables['owiLon'])
        owiSpeed = np.array(dataset.variables['owiWindSpeed'])
        owiDir = np.array(dataset.variables['owiWindDirection'])

        # Replace missing values
        owiSpeed[owiSpeed == -999.0] = np.nan
        owiDir[owiDir == -999.0] = np.nan

        # Store data
        lat_list.append(owiLat.flatten())
        lon_list.append(owiLon.flatten())
        speed_list.append(owiSpeed.flatten())
        dir_list.append(owiDir.flatten())
        datetime_list.append(pd.to_datetime(nc_file.split('-')[4], format='%Y%m%dt%H%M%S').round('h'))


max_lat = max([np.max(s) for s in lat_list])
max_lon = max([np.max(s) for s in lon_list])
min_lat = min([np.min(s) for s in lat_list])
min_lon = min([np.min(s) for s in lon_list])

# make a grid for interpolation
grid_lat = np.arange(min_lat.round(1), max_lat.round(1), 0.1)
grid_lon = np.arange(min_lon.round(1), max_lon.round(1), 0.1)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)




grid_speed_list = []
grid_dir_list = []

for lon, lat, speed, dir in tqdm(zip(lon_list, lat_list, speed_list, dir_list), total=len(lon_list)):
    # Interpolate wind speed and direction for each data set
    grid_speed = griddata((lon, lat), speed, (grid_lon, grid_lat), method='linear')
    grid_dir = griddata((lon, lat), dir, (grid_lon, grid_lat), method='linear')
    
    grid_speed_list.append(grid_speed)
    grid_dir_list.append(grid_dir)

# put data into pd.DataFrame
sar_df = pd.DataFrame({
    'datetime': datetime_list,
    'speed': grid_speed_list,
    'dir': grid_dir_list
})



#%%
# For each point in the grid, calculate the mean wind speed and direction as well
# as the number of data points contributing to that grid point
grid_speed_mean = np.nanmean(grid_speed_list, axis=0)
grid_dir_mean = np.nanmean(grid_dir_list, axis=0)
# Count the number of valid data points contributing to each grid point
grid_count = np.sum(~np.isnan(grid_speed_list), axis=0)

# Plotting
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(grid_lon, grid_lat, grid_count, cmap='viridis', shading='auto', transform=ccrs.PlateCarree())
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='Number of Data Points')
plt.title('Number of Data Points Contributing to Each Grid Point (April 2025)')
plt.show()

plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(grid_lon, grid_lat, grid_speed_mean, cmap='viridis', shading='auto', transform=ccrs.PlateCarree())
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='Wind Speed (m/s)')
plt.title('Mean Wind Speed (April 2025)')
plt.show()
# %%
from pathlib import Path
import pygrib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

CWD = Path(__file__).parent
# Replace 'your_file.grib' with the actual filename
grib_file = CWD / 'era5_wsp_and_bathymetry.grib'
start_datetime = pd.to_datetime('2025-04-01-00:00')
end_datetime = pd.to_datetime('2025-04-30-23:59')

# Open the GRIB file
grbs = pygrib.open(grib_file)

data = []
for grb in tqdm(grbs, desc="Processing GRIB messages"):
    data.append({
        'shortName': grb.shortName,
        'date': grb.date,
        'time': grb.time,
        'values': grb.values.tolist()
    })
grb_lat, grb_lon = grbs[1].latlons()
depth_lat, depth_lon = grbs[3].latlons()

df = (
    pd.DataFrame(data)
    .assign(
        datetime=lambda x: pd.to_datetime(
            x['date'].astype(str) + ' ' + x['time'].astype(str).str.zfill(4),
            format='%Y%m%d %H%M')
    )
    .loc[lambda x: (x['datetime'] >= start_datetime) & (x['datetime'] <= end_datetime)]
    .drop(columns=['date', 'time'])
)

df_u = (df[df['shortName'] == '10u'].reset_index(drop=True)
    .drop(columns=['shortName'])
    .rename(columns={'values': '10u'})
)
df_u['10u'] = df_u['10u'].apply(np.array)

df_v = (df[df['shortName'] == '10v'].reset_index(drop=True)
        .drop(columns=['shortName'])
        .rename(columns={'values': '10v'})
)
df_v['10v'] = df_v['10v'].apply(np.array)


# Combine 10u and 10v into a single DataFrame
ERA5_df = (
    pd.merge(df_u, df_v, on='datetime', how='outer')
)

ERA5_df['mag'] = ERA5_df.apply(lambda row: np.sqrt(row['10u']**2 + row['10v']**2), axis=1)
ERA5_df = ERA5_df.drop(columns=['10u', '10v'])

# Interpolate the wind speed onto the common grid (grid_lon, grid_lat)
ERA5_df['mag'] = ERA5_df['mag'].apply(
    lambda x: griddata((grb_lon.flatten(), grb_lat.flatten()), x.flatten(), (grid_lon, grid_lat), method='linear')
)




# Plotting the mean wind speed on a map
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
contour = ax.contourf(
    grid_lon, 
    grid_lat, 
    ERA5_df['mag'].mean(), 
    vmin=0,
    vmax=14,
    cmap='viridis', 
    transform=ccrs.PlateCarree()
)
ax.coastlines()
plt.colorbar(contour, ax=ax, orientation='vertical', label='Mean Wind Speed (m/s)')
plt.title('Mean Wind Speed from ERA5 Data (April 2025)')
plt.show()




combined_df = (
    pd.merge(sar_df, ERA5_df, on='datetime', how='left')
    .drop(columns=['dir'])
    .rename(columns={'speed': 'SAR_speed', 'mag': 'ERA5_speed'})
)

# For each datetime, filter the ERA5 data to only include lonlat points
# covered by the SAR data by looking at the nan values in the SAR speed
# assuming both grids are equal
combined_df['ERA5_speed'] = combined_df.apply(
    lambda row: np.where(np.isnan(row['SAR_speed']), np.nan, row['ERA5_speed']),
    axis=1
)

sar_arrays = np.stack(combined_df['SAR_speed'].values)
era5_arrays = np.stack(combined_df['ERA5_speed'].values)

sar_mean = np.nanmean(sar_arrays, axis=0)
era5_mean = np.nanmean(era5_arrays, axis=0)

plot_boundaries = {
    'min_lon': 2.4,
    'max_lon': 11.2,
    'min_lat': 40.1,
    'max_lat': 43.7
}

# Plotting the comparison of SAR and ERA5 wind speeds
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(
    grid_lon, grid_lat, sar_mean,
    cmap='viridis', shading='auto', transform=ccrs.PlateCarree(), 
    vmin=0, vmax=10
)
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='SAR Wind Speed (m/s)')
plt.title('Mean SAR Wind Speed (April 2025)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])


plt.show()
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(
    grid_lon, grid_lat, era5_mean,
    cmap='viridis', shading='auto', transform=ccrs.PlateCarree(),
    vmin=0, vmax=10
)
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='ERA5 Wind Speed (m/s)')
plt.title('Mean ERA5 Wind Speed (April 2025)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
plt.show()
# Plotting the comparison of SAR and ERA5 wind speeds
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(
    grid_lon, grid_lat, sar_mean - era5_mean,
    cmap='coolwarm', shading='auto', transform=ccrs.PlateCarree()
)
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='Difference (SAR - ERA5) Wind Speed (m/s)')
plt.title('Mean Difference in Wind Speed (SAR - ERA5) (April 2025)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
plt.show()



# %% Process bathymetry data
depth_path = CWD / 'GEBCO_17_Jun_2025_8962826f142e' / 'gebco_2024_n43.7_s40.1_w2.4_e11.2.nc'
depth_dataset = nc.Dataset(depth_path)

depth =  depth_dataset.variables['elevation'][:].filled(9999).astype(float)  # Fill NaNs with 9999 and convert to float
depth[depth>0] = np.nan  # Set positive values to NaN (land areas)

depth_lat = depth_dataset.variables['lat'][:]
depth_lon = depth_dataset.variables['lon'][:]
# create a meshgrid for the depth data
depth_lon, depth_lat = np.meshgrid(depth_lon, depth_lat)

# plot the bathymetry data
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(
    depth_lon, depth_lat, depth,
    cmap='Blues_r', shading='auto', transform=ccrs.PlateCarree(),
    vmin=-1000, vmax=0  # Adjusted for bathymetry (negative values for depth)
)
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='Bathymetry (m)')
plt.title('Bathymetry Data')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
plt.show()

# Interpolate the bathymetry data onto the SAR grid
depth_interpolated = griddata(
    (depth_lon.flatten(), depth_lat.flatten()), depth.flatten(),
    (grid_lon, grid_lat), method='linear'
)

plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(
    grid_lon, grid_lat, depth_interpolated,
    cmap='Blues_r', shading='auto', transform=ccrs.PlateCarree(),
    vmin=-1000, vmax=0  # Adjusted for bathymetry (negative values for depth)
)
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='Bathymetry (m)')
plt.title('Interpolated Bathymetry Data on SAR Grid')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
plt.show()


# %%
# Make a map that is conditionally colored based on a combination of wind speed and bathymetry
wind_speed_threshold = 5  # m/s
depth_threshold = -100  # m (depth, negative values)

sar_map = np.where(
    (sar_mean > wind_speed_threshold) & (depth_interpolated > depth_threshold),
      4, #  # Condition for high wind speed and shallow depth
      np.where(
          (sar_mean > wind_speed_threshold) & (depth_interpolated <= depth_threshold),
          3,  # Condition for high wind speed and deep water
          np.where(
              (sar_mean <= wind_speed_threshold) & (depth_interpolated > depth_threshold),
              2,  # Condition for low wind speed and shallow depth
              np.where(
                  np.isnan(sar_mean) | np.isnan(depth_interpolated),
                  np.nan,  # No data condition
                  1   # Condition for low wind speed and deep water
              )   # End of innermost where
          )
      )
)
era5_map = np.where(
    (era5_mean > wind_speed_threshold) & (depth_interpolated > depth_threshold),
      4, #  # Condition for high wind speed and shallow depth
      np.where(
          (era5_mean > wind_speed_threshold) & (depth_interpolated <= depth_threshold),
          3,  # Condition for high wind speed and deep water
          np.where(
              (era5_mean <= wind_speed_threshold) & (depth_interpolated > depth_threshold),
              2,  # Condition for low wind speed and shallow depth
              np.where(
                  np.isnan(era5_mean) | np.isnan(depth_interpolated),
                  np.nan,  # No data condition
                  1   # Condition for low wind speed and deep water
              )   # End of innermost where
          )
      )
)
color_map = {
    1: 'lightblue',  # Low wind speed and deep water
    2: 'yellow',     # Low wind speed and shallow depth
    3: 'orange',     # High wind speed and deep water
    4: 'red'         # High wind speed and shallow depth
}
# Plotting the conditional map with the defined color map
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
for value, color in color_map.items():
    plt.scatter(
        grid_lon[sar_map == value],
        grid_lat[sar_map == value],
        color=color,
        s=42,  # Adjust size for better visibility
        marker='s',  # Use square marker
        transform=ccrs.PlateCarree()
    )
ax.coastlines()
plt.title('Conditional Map Based on Wind Speed and Bathymetry (SAR)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
plt.legend(title='Conditions', loc='upper left', bbox_to_anchor=(1, 1), labels=[
    'Low Wind Speed & Deep Water',
    'Low Wind Speed & Shallow Depth',
    'High Wind Speed & Deep Water',
    'High Wind Speed & Shallow Depth'
], frameon=True)
plt.show()


plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
for value, color in color_map.items():
    plt.scatter(
        grid_lon[era5_map == value],
        grid_lat[era5_map == value],
        color=color,
        s=42,  # Adjust size for better visibility
        marker='s',  # Use square marker
        transform=ccrs.PlateCarree()
    )
ax.coastlines()
plt.title('Conditional Map Based on Wind Speed and Bathymetry (ERA5)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
plt.legend(title='Conditions', loc='upper left', bbox_to_anchor=(1, 1), labels=[
    'Low Wind Speed & Deep Water',
    'Low Wind Speed & Shallow Depth',
    'High Wind Speed & Deep Water',
    'High Wind Speed & Shallow Depth'
], frameon=True)
plt.show()
# %%
