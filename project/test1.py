from pathlib import Path
import zipfile as zf
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
from glob import glob
from scipy.interpolate import griddata
from tqdm import tqdm
import pygrib
from scipy.stats import gaussian_kde



# Your data folder with zipped SAFE files
data_folder = Path(__file__).parent / 'downloads2'
zip_files = glob(os.path.join(data_folder, '*.SAFE.zip'))

# Define the plot boundaries for the Mediterranean region (ROI)
plot_boundaries = {
    'min_lon': 2.4,
    'max_lon': 11.2,
    'min_lat': 40.1,
    'max_lat': 43.7
}

def plot_contour(plot_boundaries, grid_lat, grid_lon, wind_array,
                   title='Placeholder Title', vmin=0, vmax=10, cmap='viridis'):
    plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    mesh = plt.pcolormesh(
        grid_lon, grid_lat, wind_array,
        cmap=cmap, shading='auto', transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax
    )
    ax.coastlines()
    plt.colorbar(mesh, ax=ax, orientation='vertical', label='ERA5 Wind Speed (m/s)')
    plt.title(title)
    plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
    plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])    
    plt.show()

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
grid_lat = np.arange(min_lat.round(1), max_lat.round(1), 0.02)
grid_lon = np.arange(min_lon.round(1), max_lon.round(1), 0.02)
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
plt.figure(figsize=(12, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(grid_lon, grid_lat, grid_count, cmap='Greens', shading='auto', transform=ccrs.PlateCarree())
rect = patches.Rectangle(
    (plot_boundaries['min_lon'], plot_boundaries['min_lat']),
    plot_boundaries['max_lon'] - plot_boundaries['min_lon'],
    plot_boundaries['max_lat'] - plot_boundaries['min_lat'],
    linewidth=2, edgecolor='black', facecolor='none', transform=ccrs.PlateCarree()
)
ax.coastlines()
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', label='Number of Data Points')
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Number of Data Points', fontsize=16)
ax.add_patch(rect)
plt.show()

plt.figure(figsize=(12, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(grid_lon, grid_lat, grid_speed_mean, cmap='viridis',
                      shading='auto', transform=ccrs.PlateCarree(),
                      vmin=0, vmax=10)
ax.coastlines()
plt.colorbar(mesh, ax=ax, orientation='vertical', label='Wind Speed (m/s)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
plt.show()
# %%
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

ERA5_df['mag_unsmoothed'] = ERA5_df['mag'].copy()
plot_contour(plot_boundaries, grb_lat, grb_lon, ERA5_df['mag_unsmoothed'].mean(),
               title='Mean Wind Speed from ERA5 Data (April 2025)',
               vmin=0, vmax=10, cmap='viridis')

# Interpolate the wind speed onto the common grid (grid_lon, grid_lat)
ERA5_df['mag'] = ERA5_df['mag'].apply(
    lambda x: griddata((grb_lon.flatten(), grb_lat.flatten()), x.flatten(), (grid_lon, grid_lat), method='linear')
)




# Plotting the mean wind speed on a map
plot_contour(plot_boundaries, grid_lat, grid_lon, ERA5_df['mag'].mean(),
               title='Mean Wind Speed from ERA5 Data (April 2025)',
                vmin=0, vmax=10, cmap='viridis')


combined_df = (
    pd.merge(sar_df, ERA5_df, on='datetime', how='left')
    .drop(columns=['dir'])
    .rename(columns={'speed': 'SAR_speed', 'mag': 'ERA5_speed'})
    .assign(hour=lambda x: x['datetime'].dt.hour)
)

# For each datetime, filter the ERA5 data to only include lonlat points
# covered by the SAR data by looking at the nan values in the SAR speed
# assuming both grids are equal
combined_df['ERA5_speed'] = combined_df.apply(
    lambda row: np.where(np.isnan(row['SAR_speed']), np.nan, row['ERA5_speed']),
    axis=1
)

sar_arrays = np.stack(combined_df['SAR_speed'].values)
sar_ascending_arrays = np.stack(combined_df.loc[lambda x: x['hour'].isin([17,18])]['SAR_speed'].values)
sar_descending_arrays = np.stack(combined_df.loc[lambda x: x['hour'].isin([5,6])]['SAR_speed'].values)
# Filter out NaN values from the SAR data
era5_arrays = np.stack(combined_df['ERA5_speed'].values)
era5_ascending_arrays = np.stack(combined_df.loc[lambda x: x['hour'].isin([17,18])]['ERA5_speed'].values)
era5_descending_arrays = np.stack(combined_df.loc[lambda x: x['hour'].isin([5,6])]['ERA5_speed'].values)

sar_mean = np.nanmean(sar_arrays, axis=0)
sar_ascending_mean = np.nanmean(sar_ascending_arrays, axis=0)
sar_descending_mean = np.nanmean(sar_descending_arrays, axis=0)
era5_mean = np.nanmean(era5_arrays, axis=0)
era5_ascending_mean = np.nanmean(era5_ascending_arrays, axis=0)
era5_descending_mean = np.nanmean(era5_descending_arrays, axis=0)

sar_mask = ~np.isnan(sar_mean)
ERA5_df['mag_masked'] = ERA5_df['mag'].apply(
    lambda x: np.where(sar_mask, x, np.nan)
)
era5_arrays_full = np.stack(ERA5_df['mag_masked'].values)
era5_mean_full = np.nanmean(era5_arrays_full, axis=0)
# Plot the masked mean wind speed from ERA5


plot_contour(plot_boundaries, grid_lat, grid_lon, ERA5_df['mag_masked'].mean(),
               title='Mean ERA5 Wind Speed (April 2025) - Masked by SAR Data')

# Plotting the comparison of SAR and ERA5 wind speeds
plot_contour(plot_boundaries, grid_lat, grid_lon, sar_mean,
               title='Mean SAR Wind Speed (April 2025)')
plot_contour(plot_boundaries, grid_lat, grid_lon, sar_ascending_mean,
               title='Mean SAR Wind Speed (Ascending Passes, April 2025)')
plot_contour(plot_boundaries, grid_lat, grid_lon, sar_descending_mean,
               title='Mean SAR Wind Speed (Descending Passes, April 2025)')
plot_contour(plot_boundaries, grid_lat, grid_lon, era5_mean,
               title='Mean ERA5 Wind Speed (April 2025)')
plot_contour(plot_boundaries, grid_lat, grid_lon, era5_ascending_mean,
               title='Mean ERA5 Wind Speed (Ascending Passes, April 2025)')
plot_contour(plot_boundaries, grid_lat, grid_lon, era5_descending_mean,
               title='Mean ERA5 Wind Speed (Descending Passes, April 2025)')    


# Plotting the comparison of SAR and ERA5 wind speeds
plot_contour(plot_boundaries, grid_lat, grid_lon, sar_mean - era5_mean,
               title='Mean Difference in Wind Speed (SAR - ERA5) (April 2025)',
               vmin=-3, vmax=3, cmap='RdBu_r')

plot_contour(plot_boundaries, grid_lat, grid_lon, era5_mean - era5_mean_full,
               title='Mean Difference in Wind Speed (ERA5 - ERA5 Full Month) (April 2025)',
               vmin=-3, vmax=3, cmap='RdBu_r')

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
plot_contour(plot_boundaries, depth_lat, depth_lon, depth,
               title='Bathymetry Data (GEBCO)', vmin=-1000, vmax=0, cmap='Blues_r')


# Interpolate the bathymetry data onto the SAR grid
depth_interpolated = griddata(
    (depth_lon.flatten(), depth_lat.flatten()), depth.flatten(),
    (grid_lon, grid_lat), method='linear'
)
plot_contour(plot_boundaries, grid_lat, grid_lon, depth_interpolated,
               title='Interpolated Bathymetry Data on SAR Grid',
               vmin=-1000, vmax=0, cmap='Blues_r')


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
era5_map_full = np.where(
    (era5_mean_full > wind_speed_threshold) & (depth_interpolated > depth_threshold),
      4, #  # Condition for high wind speed and shallow depth
      np.where(
          (era5_mean_full > wind_speed_threshold) & (depth_interpolated <= depth_threshold),
          3,  # Condition for high wind speed and deep water
          np.where(
              (era5_mean_full <= wind_speed_threshold) & (depth_interpolated > depth_threshold),
              2,  # Condition for low wind speed and shallow depth
              np.where(
                  np.isnan(era5_mean_full) | np.isnan(depth_interpolated),
                  np.nan,  # No data condition
                  1   # Condition for low wind speed and deep water
              )   # End of innermost where
          )
      )
)
color_map = {
    1: 'lightblue',  # Low wind speed and deep water
    2: 'yellow',     # Low wind speed and shallow depth
    3: 'sandybrown',     # High wind speed and deep water
    4: 'lightgreen'         # High wind speed and shallow depth
}
ListedColormap = plt.cm.colors.ListedColormap([color_map[i] for i in range(1, 5)])

# Plotting the conditional map 
def plot_conditional_map(plot_boundaries, grid_lat, grid_lon, sar_map, ListedColormap,
                         title='Conditional Map Based on Wind Speed and Bathymetry'):
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    total_points = np.count_nonzero(~np.isnan(sar_map))
    cond1_count = np.count_nonzero(sar_map == 1)
    cond2_count = np.count_nonzero(sar_map == 2)
    cond3_count = np.count_nonzero(sar_map == 3)
    cond4_count = np.count_nonzero(sar_map == 4)
    mesh = plt.pcolormesh(
        grid_lon, grid_lat, sar_map,
        cmap=ListedColormap,
        shading='auto', transform=ccrs.PlateCarree(),
        vmin=0.5, vmax=4.5)
    ax.coastlines()
    plt.title(title)
    plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
    plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', label='Conditions')
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels([
    f'Low Wind Speed\n& Deep Water ({cond1_count/total_points:.1%})',
    f'Low Wind Speed\n& Shallow Depth ({cond2_count/total_points:.1%})',
    f'High Wind Speed\n& Deep Water ({cond3_count/total_points:.1%})',
    f'High Wind Speed\n& Shallow Depth ({cond4_count/total_points:.1%})'
])
    plt.tight_layout()
    plt.show()

plot_conditional_map(plot_boundaries, grid_lat, grid_lon, sar_map, ListedColormap)

# Plotting the conditional map for ERA5 data
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(
    grid_lon, grid_lat, era5_map,
    cmap=ListedColormap,
    shading='auto', transform=ccrs.PlateCarree(),
    vmin=0.5, vmax=4.5
)
ax.coastlines()
plt.title('Conditional Map Based on Wind Speed and Bathymetry (ERA5)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', label='Conditions')
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels([
    'Low Wind Speed\n& Deep Water',
    'Low Wind Speed\n& Shallow Depth',
    'High Wind Speed\n& Deep Water',
    'High Wind Speed\n& Shallow Depth'
])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = plt.pcolormesh(
    grid_lon, grid_lat, era5_map_full,
    cmap=ListedColormap,
    shading='auto', transform=ccrs.PlateCarree(),
    vmin=0.5, vmax=4.5
)
ax.coastlines()
plt.title('Conditional Map Based on Wind Speed and Bathymetry (ERA5 full month)')
plt.xlim(plot_boundaries['min_lon'], plot_boundaries['max_lon'])
plt.ylim(plot_boundaries['min_lat'], plot_boundaries['max_lat'])
cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', label='Conditions')
cbar.set_ticks([1, 2, 3, 4])
cbar.set_ticklabels([
    'Low Wind Speed\n& Deep Water',
    'Low Wind Speed\n& Shallow Depth',
    'High Wind Speed\n& Deep Water',
    'High Wind Speed\n& Shallow Depth'
])
plt.tight_layout()
plt.show()
# %% Look for temporal patterns in the ERA5 data
# Create a DataFrame for the ERA5 data with datetime as index
era5_df = pd.DataFrame({
    'datetime': ERA5_df['datetime'],
    'wind_speed': ERA5_df['mag'],
    'mean_wind_speed': ERA5_df['mag'].apply(lambda x: np.nanmean(x) if isinstance(x, np.ndarray) else np.nan)
}).assign(hour=lambda x: x['datetime'].dt.hour)
era5_df.set_index('datetime', inplace=True)

# plot the mean wind speed for each hour of the day
plt.figure(figsize=(12, 6))
era5_hourly_mean = era5_df.groupby('hour')['mean_wind_speed'].mean()
era5_hourly_mean.plot(kind='bar', color='skyblue')
plt.axhline(np.nanmean(era5_hourly_mean), color='red', linestyle='--', label='Overall Mean')
plt.legend()
plt.title('Mean Wind Speed by Hour of the Day (ERA5)')
plt.xlabel('Hour of the Day')
plt.ylabel('Mean Wind Speed (m/s)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


ERA5_df['hour'] = ERA5_df['datetime'].dt.hour
hourly_mag_fields = ERA5_df.groupby('hour')['mag'].apply(lambda x: np.nanmean(np.stack(x), axis=0) if isinstance(x.iloc[0], np.ndarray) else np.nan)

correction_factors = {
    hour: hourly_mag_fields[hour] / era5_mean_full
    for hour in range(24)
}

sar_corrected = (
    sar_df.copy()
    .assign(
        hour=lambda x: x['datetime'].dt.hour,
        correction=lambda x: x['hour'].map(correction_factors)
    )
)
sar_corrected['corrected_speed'] = sar_corrected.apply(
    lambda row: row['speed'] * row['correction'] if isinstance(row['speed'], np.ndarray) else np.nan,
    axis=1
)
corrected_arrays = np.stack(sar_corrected['corrected_speed'].values)
sar_corrected_mean = np.nanmean(corrected_arrays, axis=0)
plot_contour(plot_boundaries, grid_lat, grid_lon, sar_corrected_mean,
               title='Mean Corrected SAR Wind Speed (April 2025)')

plot_contour(plot_boundaries, grid_lat, grid_lon, sar_corrected_mean - sar_mean,
                title='Mean Difference in Corrected SAR Wind Speed (April 2025)',
                vmin=-1, vmax=1, cmap='RdBu_r')


bias = era5_mean_full - era5_mean
sar_without_bias = sar_mean - bias
plot_contour(plot_boundaries, grid_lat, grid_lon, sar_without_bias,
               title='Mean SAR Wind Speed without ERA5 Bias (April 2025)',
               vmin=0, vmax=10, cmap='viridis')
plot_contour(plot_boundaries, grid_lat, grid_lon, sar_without_bias - era5_mean,
               title='Mean Difference in SAR Wind Speed without ERA5 Bias (April 2025)',
               vmin=-3, vmax=3, cmap='RdBu_r')




# %%


errors1 = sar_mean - era5_mean
errors2 = sar_corrected_mean - era5_mean
errors3 = sar_corrected_mean - sar_mean
errors4 = sar_without_bias - era5_mean
# Plot error distributions as PDFs using kernel density estimation (KDE)
plt.figure(figsize=(12, 6))

# Remove NaNs for KDE
err1 = errors1.flatten()
err2 = errors2.flatten()
err3 = errors3.flatten()
err4 = errors4.flatten()
err1 = err1[~np.isnan(err1)]
err2 = err2[~np.isnan(err2)]
err3 = err3[~np.isnan(err3)]
err4 = err4[~np.isnan(err4)]

# KDE for both error arrays
kde1 = gaussian_kde(err1)
kde2 = gaussian_kde(err2)
kde3 = gaussian_kde(err3)
kde4 = gaussian_kde(err4)
x_min = min(err1.min(), err2.min(), err3.min(), err4.min())
x_max = max(err1.max(), err2.max(), err3.max(), err4.max())
x_grid = np.linspace(x_min, x_max, 500)

plt.plot(x_grid, kde1(x_grid), label='SAR - ERA5', color='blue')
plt.plot(x_grid, kde2(x_grid), label='Corrected SAR - ERA5', color='orange')
plt.plot(x_grid, kde3(x_grid), label='Corrected SAR - SAR', color='green')
plt.plot(x_grid, kde4(x_grid), label='Corrected SAR - SAR without Bias', color='red')
plt.xlabel('Error (m/s)')
plt.ylabel('Probability Density')
plt.title('Error Distributions (PDF): SAR vs ERA5 and Corrected SAR vs ERA5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

