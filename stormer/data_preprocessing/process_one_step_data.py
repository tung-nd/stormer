import os
import argparse
import numpy as np
import xarray as xr
import h5py
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


def create_one_step_dataset(root_dir, save_dir, split, years, list_vars, chunk_size=None):
    save_dir_split = os.path.join(save_dir, split)
    os.makedirs(save_dir_split, exist_ok=True)
    
    list_constant_vars = [v for v in list_vars if v in CONSTANTS]
    list_single_vars = [v for v in list_vars if v in SINGLE_LEVEL_VARS and v not in CONSTANTS]
    list_pressure_vars = [v for v in list_vars if v in PRESSURE_LEVEL_VARS]
    
    # load a constant variable to save lat and lon arrays
    ds_constant = xr.open_dataset(os.path.join(root_dir, f'{list_constant_vars[0]}.nc'))
    lat = ds_constant.latitude.to_numpy()
    lat.sort()
    lon = ds_constant.longitude.to_numpy()
    lon.sort()
    np.save(os.path.join(save_dir, 'lat.npy'), lat)
    np.save(os.path.join(save_dir, 'lon.npy'), lon)
    
    for year in tqdm(years, desc='years', position=0):
        ds_sample = xr.open_dataset(os.path.join(root_dir, list_single_vars[0], f'{year}.nc'))
        if chunk_size is not None:
            n_chunks = len(ds_sample.time) // chunk_size + 1
        else:
            n_chunks = 1
            chunk_size = len(ds_sample.time)
        
        idx_in_year = 0
        
        ds_dict = {}
        for var in (list_single_vars + list_pressure_vars):
            ds_dict[var] = xr.open_dataset(os.path.join(root_dir, var, f'{year}.nc'))

        for chunk_id in tqdm(range(n_chunks), desc='chunks', position=1, leave=False):
            dict_np = {}
            list_time_stamps = None
            ### convert ds to numpy
            for var in (list_single_vars + list_pressure_vars):
                ds = ds_dict[var].isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                if list_time_stamps is None:
                    list_time_stamps = ds.time.values
                if var in list_single_vars:
                    dict_np[var] = ds[var].values
                else:
                    available_levels = ds.level.values
                    ds_np = ds[var].values
                    for i, level in enumerate(available_levels):
                        if level in DEFAULT_PRESSURE_LEVELS:
                            dict_np[f'{var}_{level}'] = ds_np[:, i]
                    
            for i in tqdm(range(len(list_time_stamps)), desc='time stamps', position=2, leave=False):
                data_dict = {
                    'input': {'time': str(list_time_stamps[i])}
                }
                for var in dict_np.keys():
                    data_dict['input'][var] = dict_np[var][i]
                for var in list_constant_vars:
                    constant_path = os.path.join(root_dir, f'{var}.nc')
                    constant_field = xr.open_dataset(constant_path)[var].to_numpy()
                    constant_field = constant_field.reshape(constant_field.shape[-2:])
                    data_dict['input'][var] = constant_field
                    
                with h5py.File(os.path.join(save_dir_split, f'{year}_{idx_in_year:04}.h5'), 'w', libver='latest') as f:
                    for main_key, sub_dict in data_dict.items():
                        # Create a group for the main key (e.g., 'input' or 'output')
                        group = f.create_group(main_key)
                        
                        # Now, save each array in the sub-dictionary to this group
                        for sub_key, array in sub_dict.items():
                            if sub_key != 'time':
                                group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                            else:
                                group.create_dataset(sub_key, data=array, compression=None)
                
                idx_in_year += 1


def parse_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--start_year', type=int, default=1979, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=2019, help='End year for the data range.')
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset (train, val, test).")
    parser.add_argument("--chunk_size", type=int, default=10, help="Chunk size for reading datasets (default=10).")
    
    return parser.parse_args()


def main():
    args = parse_args()

    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        split=args.split,
        years=list(range(args.start_year, args.end_year)),
        list_vars=VARS,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()