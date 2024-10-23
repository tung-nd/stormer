import os
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
from stormer.utils.data_utils import (
    CONSTANTS,
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS
)

# change as needed
VARS = [
    "anisotropy_of_sub_gridscale_orography",
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    "skin_temperature",
    "snow_depth",
    
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "mean_sea_level_pressure",
    "sea_ice_cover",
    "sea_surface_temperature",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "toa_incident_solar_radiation_6hr",
    "toa_incident_solar_radiation_12hr",
    "toa_incident_solar_radiation_24hr",
    "total_cloud_cover",
    "total_precipitation_6hr",
    "total_precipitation_12hr",
    "total_precipitation_24hr",
    "total_column_water_vapour",
    
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "wind_speed"
]

def parse_args():
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--start_year', type=int, default=1979, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=2021, help='End year for the data range.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for reading datasets (default=100).')
    parser.add_argument('--lead_time', type=int, default=None, help='Lead time for difference normalization.')  # None means we're computing normalization for the original data
    parser.add_argument('--data_frequency', type=int, default=6, help='Data frequency in hours (default=6).')  # this depends on the dataset
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
    start_year = args.start_year
    end_year = args.end_year
    chunk_size = args.chunk_size
    lead_time = args.lead_time
    data_freq = args.data_frequency
    
    years = list(range(start_year, end_year + 1))
    os.makedirs(save_dir, exist_ok=True)

    list_constant_vars = [v for v in VARS if v in CONSTANTS]
    list_single_vars = [v for v in VARS if v in SINGLE_LEVEL_VARS and v not in CONSTANTS]
    list_pressure_vars = [v for v in VARS if v in PRESSURE_LEVEL_VARS]
    
    mean_file_name = f"normalize_diff_mean_{lead_time}.npz" if lead_time != -1 else "normalize_mean.npz"
    std_file_name = f"normalize_diff_std_{lead_time}.npz" if lead_time != -1 else "normalize_std.npz"

    # initialize normalization values if not exist, else load them
    if not os.path.exists(os.path.join(save_dir, mean_file_name)):
        normalize_mean = {}
        normalize_std = {}

        for var in list_single_vars:
            normalize_mean[var] = []
            normalize_std[var] = []
        for var in list_pressure_vars:
            for level in DEFAULT_PRESSURE_LEVELS:
                normalize_mean[f'{var}_{level}'] = []
                normalize_std[f'{var}_{level}'] = []
    else:
        normalize_mean = np.load(os.path.join(save_dir, mean_file_name))
        normalize_std = np.load(os.path.join(save_dir, std_file_name))
        normalize_mean = {k: list(v) for k, v in normalize_mean.items()}
        normalize_std = {k: list(v) for k, v in normalize_std.items()}

    steps = lead_time // data_freq if lead_time else None

    for var in tqdm(list_single_vars + list_pressure_vars, desc='variables', position=0):
        for year in tqdm(years, desc='years', position=1, leave=False):
            path = os.path.join(root_dir, var, f'{year}.nc')
            ds = xr.open_dataset(path)
            
            # chunk to smaller sizes
            if chunk_size is not None:
                n_chunks = len(ds.time) // chunk_size + 1
            else:
                n_chunks = 1
                chunk_size = len(ds.time)
            
            for chunk_id in tqdm(range(n_chunks), desc='chunks', position=2, leave=False):
                ds_small = ds.isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                if var in SINGLE_LEVEL_VARS:
                    ds_np = ds_small[var].values # N, H, W
                    if steps is not None:
                        ds_np = ds_np[steps:] - ds_np[:-steps]
                    normalize_mean[var].append(np.nanmean(ds_np))
                    normalize_std[var].append(np.nanstd(ds_np))
                else:
                    ds_np = ds_small[var].values # N, Levels, H, W
                    levels_in_ds = ds.level.values
                    assert np.sum(np.array(DEFAULT_PRESSURE_LEVELS) - levels_in_ds) == 0 # ensure the same order of pressure levels
                    for i, level in enumerate(levels_in_ds):
                        ds_np_lev = ds_np[:, i]
                        if steps is not None:
                            ds_np_lev = ds_np_lev[steps:] - ds_np_lev[:-steps]
                        normalize_mean[f'{var}_{level}'].append(np.nanmean(ds_np_lev))
                        normalize_std[f'{var}_{level}'].append(np.nanstd(ds_np_lev))
            
        if var in SINGLE_LEVEL_VARS:
            mean_over_files, std_over_files = np.array(normalize_mean[var]), np.array(normalize_std[var])
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean_over_files.mean()
            normalize_mean[var] = mean.reshape([1])
            normalize_std[var] = std.reshape([1])
        
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)
        else:
            for l in DEFAULT_PRESSURE_LEVELS:
                var_lev = f'{var}_{l}'
                mean_over_files, std_over_files = np.array(normalize_mean[var_lev]), np.array(normalize_std[var_lev])
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean_over_files.mean()
                normalize_mean[var_lev] = mean.reshape([1])
                normalize_std[var_lev] = std.reshape([1])
        
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)
        
    for var in list_constant_vars:
        if steps is not None:
            normalize_mean[var] = [0.0]
            normalize_std[var] = [0.0]
        else:
            path = os.path.join(root_dir, f'{var}.nc')
            ds = xr.open_dataset(path)
            ds_np = ds[var].values
            normalize_mean[var] = ds_np.mean().reshape([1])
            normalize_std[var] = ds_np.std().reshape([1])

    np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
    np.savez(os.path.join(save_dir, std_file_name), **normalize_std)