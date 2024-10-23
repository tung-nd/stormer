import os
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm
import regridding

# change as needed
VARS = [
    # constants
    "angle_of_sub_gridscale_orography.nc",
    "geopotential_at_surface.nc",
    "high_vegetation_cover.nc",
    "lake_cover.nc",
    "lake_depth.nc",
    "land_sea_mask.nc",
    "low_vegetation_cover.nc",
    "slope_of_sub_gridscale_orography.nc",
    "soil_type.nc",
    "standard_deviation_of_filtered_subgrid_orography.nc",
    "standard_deviation_of_orography.nc",
    "type_of_high_vegetation.nc",
    "type_of_low_vegetation.nc",

    # surface variables
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "10m_wind_speed",
    "mean_sea_level_pressure",

    # pressure level variables
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

def parse_args():
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--ddeg_out', type=float, default=1.40625, help='Output grid spacing in degrees.')
    parser.add_argument('--start_year', type=int, default=1979, help='Start year for the data range.')
    parser.add_argument('--end_year', type=int, default=2021, help='End year for the data range.')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for reading datasets (default=100).')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir
    ddeg_out = args.ddeg_out
    start_year = args.start_year
    end_year = args.end_year
    chunk_size = args.chunk_size
    
    years = list(range(start_year, end_year + 1))
    os.makedirs(save_dir, exist_ok=True)

    lat_start = -90 + ddeg_out / 2
    lat_stop = 90 - ddeg_out / 2
    new_lat = np.linspace(lat_start, lat_stop, num=int(180/ddeg_out), endpoint=True)
    new_lon = np.linspace(0, 360, num=int(360//ddeg_out), endpoint=False)
    
    regridder = None
    var_dirs = [os.path.join(root_dir, v) for v in VARS]
    
    for dir in tqdm(var_dirs, desc='vars', position=0):
        var_name = os.path.basename(dir)
        
        if '.nc' in dir:
            ds_in = xr.open_dataset(dir).transpose(..., 'latitude', 'longitude')
            
            if regridder is None:
                old_lon = ds_in.coords['longitude'].data
                old_lat = ds_in.coords['latitude'].data
                source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
                target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
                regridder = regridding.ConservativeRegridder(source_grid, target_grid)
                
            ds_out = regridder.regrid_dataset(ds_in)
            ds_out.to_netcdf(os.path.join(save_dir, var_name))
        
        else:
            os.makedirs(os.path.join(save_dir, var_name), exist_ok=True)
            for year in tqdm(years, desc='years', position=1, leave=False):
                ds_in = xr.open_dataset(os.path.join(dir, f'{year}.nc'), chunks={'time': chunk_size})
                
                if regridder is None:
                    old_lon = ds_in.coords['longitude'].data
                    old_lat = ds_in.coords['latitude'].data
                    source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
                    target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
                    regridder = regridding.ConservativeRegridder(source_grid, target_grid)
                
                ds_out = regridder.regrid_dataset(ds_in).transpose(..., 'latitude', 'longitude')
                ds_out.to_netcdf(os.path.join(save_dir, var_name, f'{year}.nc'))

if __name__ == "__main__":
    main()
