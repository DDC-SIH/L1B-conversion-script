import h5py
import numpy as np
import cartopy.crs as ccrs
from osgeo import gdal
import os
import matplotlib.pyplot as plt

class L1BProcessor:
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        self.band_fills = {
            'VIS': {'IMG_VIS': 0, 'IMG_SWIR': 0},
            'IR': {'IMG_MIR': 1023, 'IMG_TIR1': 1023, 'IMG_TIR2': 1023},
            'WV': {'IMG_WV': 1023}
        }
        self.globe = ccrs.Globe(
            semimajor_axis=6378137.0,
            semiminor_axis=6356752.31414
        )
        self.geos_proj = ccrs.Geostationary(
            central_longitude=74.0,
            satellite_height=35752815.622,
            false_easting=0,
            false_northing=0,
            sweep_axis='x',
            globe=self.globe
        )

    def find_bounding_square(self, data, fill_value):
        """Find valid data bounds from center outwards"""
        rows, cols = data.shape
        center_y, center_x = rows // 2, cols // 2
        
        # Scan from center in all directions
        top = center_y
        while top > 0 and data[top, center_x] != fill_value:
            top -= 1
        top += 1  # Adjust to last valid pixel
        
        bottom = center_y
        while bottom < rows - 1 and data[bottom, center_x] != fill_value:
            bottom += 1
        bottom -= 1
        
        left = center_x
        while left > 0 and data[center_y, left] != fill_value:
            left -= 1
        left += 1
        
        right = center_x
        while right < cols - 1 and data[center_y, right] != fill_value:
            right += 1
        right -= 1
        
        return {
            "top_left": (top, left),
            "top_right": (top, right),
            "bottom_left": (bottom, left),
            "bottom_right": (bottom, right)
        }

    def count_cold_space(self):
        """Count cold space pixels for each band"""
        cold_space_counts = {}
        with h5py.File(self.h5_file_path, 'r') as h5f:
            for band_type, bands in self.band_fills.items():
                cold_space_counts[band_type] = {}
                for band_name, fill_value in bands.items():
                    if band_name in h5f:
                        data = h5f[band_name][:]
                        if len(data.shape) == 3:
                            data = data[0]
                        cold_count = np.sum(data == fill_value)
                        valid_bounds = self.find_bounding_square(data, fill_value)
                        cold_space_counts[band_type][band_name] = {
                            'total_cold_pixels': cold_count,
                            'bounds': valid_bounds
                        }
        return cold_space_counts

    def read_clean_data(self):
        """Read and clean data excluding cold space"""
        clean_data = {}
        with h5py.File(self.h5_file_path, 'r') as h5f:
            for band_type, bands in self.band_fills.items():
                for band_name, fill_value in bands.items():
                    if band_name in h5f:
                        data = h5f[band_name][:]
                        if len(data.shape) == 3:
                            data = data[0]
                            
                        # Get valid bounds
                        bounds = self.find_bounding_square(data, fill_value)
                        top = bounds['top_left'][0]
                        bottom = bounds['bottom_left'][0]
                        left = bounds['top_left'][1]
                        right = bounds['top_right'][1]
                        
                        # Subset data using bounds
                        subset = data[top:bottom+1, left:right+1]
                        mask = subset != fill_value
                        clean_data[band_name] = np.ma.masked_array(subset, ~mask)
        return clean_data

    def compute_resolution(self, array_size):
        """Compute resolution parameters"""
        satellite_height = 35752815.622   
        fov_degrees = 17.989905
        fov_radians = np.radians(fov_degrees)
        
        angular_resolution = fov_radians / array_size
        nadir_resolution = satellite_height * angular_resolution
        
        return {
            'angular_resolution_km': np.degrees(angular_resolution),
            'nadir_resolution_km': nadir_resolution/1000
        }

    def create_cog(self, input_tiff, output_tiff):
        """Convert GeoTIFF to Cloud Optimized GeoTIFF"""
        cog_driver = gdal.GetDriverByName('COG')
        if cog_driver is None:
            # Fallback method using translate if COG driver is not available
            translate_options = gdal.TranslateOptions(
                format='COG',
                creationOptions=[
                    'COMPRESS=DEFLATE',
                    'PREDICTOR=2',
                    'BLOCKSIZE=512',
                    'OVERVIEW_RESAMPLING=AVERAGE'
                ]
            )
            gdal.Translate(output_tiff, input_tiff, options=translate_options)
        else:
            # Use COG driver directly if available
            ds = gdal.Open(input_tiff)
            cog_driver.CreateCopy(
                output_tiff, 
                ds, 
                options=[
                    'COMPRESS=DEFLATE',
                    'PREDICTOR=2',
                    'BLOCKSIZE=512',
                    'OVERVIEW_RESAMPLING=AVERAGE'
                ]
            )
            ds = None
        return output_tiff

    def save_geotiff(self, data, output_path, band_name):
        """Save data as Cloud Optimized GeoTIFF with proper projections"""
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data.shape
        
        # Get extent from geostationary projection
        extent = self.geos_proj.x_limits + self.geos_proj.y_limits
        left, right = extent[:2]
        bottom, top = extent[2:]
        
        # Create initial file
        temp_tiff = f"temp_{band_name}_geos.tif"
        ds = driver.Create(temp_tiff, cols, rows, 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(data)
        
        # Set projection from cartopy
        wkt = self.geos_proj.to_wkt()
        ds.SetProjection(wkt)
        
        # Calculate pixel size and set geotransform
        pixel_x = (right - left) / cols
        pixel_y = (top - bottom) / rows
        ds.SetGeoTransform([left, pixel_x, 0, top, 0, -pixel_y])
        ds = None
        
        # First translate to intermediate projection
        translated_tiff = f"translated_{band_name}.tif"
        gdal.Translate(translated_tiff, temp_tiff)
        
        # Create intermediate WGS84 GeoTIFF
        intermediate_wgs84 = f"{output_path}/{band_name}_wgs84_temp.tif"
        warp_options = gdal.WarpOptions(
            format='GTiff',
            dstSRS='EPSG:4326',
            xRes=0.036,
            yRes=0.036,
            resampleAlg=gdal.GRA_Bilinear
        )
        gdal.Warp(intermediate_wgs84, translated_tiff, options=warp_options)

        # Convert to COG
        output_file = f"{output_path}/{band_name}_wgs84_cog.tif"
        self.create_cog(intermediate_wgs84, output_file)
        
        # Cleanup temporary files
        os.remove(temp_tiff)
        os.remove(translated_tiff)
        os.remove(intermediate_wgs84)
        
        return output_file

    def process_all(self, output_dir):
        """Process all bands"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Count cold space
        cold_counts = self.count_cold_space()
        print("Cold Space Counts:")
        print(cold_counts)
        
        # Read and clean data
        clean_data = self.read_clean_data()
        
        # Process each band
        for band_name, data in clean_data.items():
            try:
                output_file = self.save_geotiff(data, output_dir, band_name)
                print(f"Processed {band_name} to: {output_file}")
            except Exception as e:
                print(f"Error processing {band_name}: {str(e)}")

# Usage
if __name__ == "__main__":
    h5_file = "3RIMG_04SEP2024_1015_L1B_STD_V01R00.h5"
    output_dir = "output_geotiffs"
    
    processor = L1BProcessor(h5_file)
    processor.process_all(output_dir)