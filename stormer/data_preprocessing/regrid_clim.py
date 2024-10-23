import numpy as np
import xarray as xr
import argparse
import regridding

def parse_args():
    parser = argparse.ArgumentParser(description='Regridding climatology NetCDF file.')
    parser.add_argument('--input_file', type=str, required=True, help='Input NetCDF file.')
    parser.add_argument('--output_file', type=str, required=True, help='Output NetCDF file.')
    parser.add_argument('--ddeg_out', type=float, required=True, help='Output grid spacing in degrees.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    ddeg_out = args.ddeg_out
    
    # Load the input NetCDF file
    clim = xr.open_dataset(input_file).astype(np.float32)
    clim = clim.sortby('latitude')
    
    # Get the latitude and longitude from the input dataset
    lat = clim.latitude.values.astype(np.float32)
    lon = clim.longitude.values.astype(np.float32)
    
    # Calculate the new latitude and longitude grids based on ddeg_out
    lat_start = -90 + ddeg_out / 2
    lat_stop = 90 - ddeg_out / 2
    new_lat = np.linspace(lat_start, lat_stop, num=int(180/ddeg_out), endpoint=True)
    new_lon = np.linspace(0, 360, num=int(360//ddeg_out), endpoint=False)
    
    # Create source and target grids for regridding
    source_grid = regridding.Grid.from_degrees(lat=np.sort(lat), lon=lon)
    target_grid = regridding.Grid.from_degrees(lat=new_lat, lon=new_lon)
    
    # Create the regridder
    regridder = regridding.ConservativeRegridder(source_grid, target_grid)
    
    print('Start regridding')
    
    # Perform the regridding
    clim_new = regridder.regrid_dataset(clim)
    
    # Save the regridded dataset to the output file
    clim_new.to_netcdf(output_file)

if __name__ == "__main__":
    main()
