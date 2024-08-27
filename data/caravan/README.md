![](caravan-long-logo.png)

**Caravan** is an open community dataset of meteorological forcing data, catchment attributes, and discharge data for catchments around the world. Additionally, Caravan provides code to derive meteorological forcing data and catchment attributes from the same data sources in the cloud, making it easy for anyone to extend Caravan to new catchments. The vision of Caravan is to provide the foundation for a truly global open source community resource that will grow over time. 

**Note: This is a pre-release of the Caravan dataset, which is currently under revision.**

## Dataset structure

The dataset is structured as follows:

- `attributes/`: This folder contains one subfolder per source dataset, which contains three csv (comma separated values) files. One contains the attributes derived from HydroATLAS, called `attributes_hydroatlas_{source}.csv`, the other contains the climate indices derived from ERA5-Land time , called `attributes_caravan_{source}.csv`, the last contains metadata information (such as gauge location and name), called `attributes_other_{source}.csv`, where `{source}` is:
    - `camels` for CAMELS (US)
    - `camelsaus` for CAMELS-AUS
    - `camelsbr` for CAMELS-BR
    - `camelscl` for CAMELS-CL
    - `camelsgb` for CAMELS-GB
    - `hysets` for HYSETS
    - `lamah` for LamaH-CE
The first column in all attributes file is called `gauge_id` and contains a unique basin identifier of the form `{source}_{id}`, where `{id}` is the basin id as defined in the original source dataset.

- `timeseries/`: This folder contains two subfolders, `csv/` and `netcdf/`, that both share the same structure and contain the same data, once as csv-files and once as netCDF files. Each of these two subfolders contains one subfolder per source dataset, where the name of the subfolder matches the `{source}` in the attributes csv-file. Within these source dataset specific subdirectories, there is one file (either csv or netcdf) per basin, containing all time series data (meteorological forcings, state variables and discharge). The netCDF files also contain metadata information, namely the units of each time series feature, the timezone of the data, and information on the data sources (see also below for the same information).

- `shapefiles/`: This folder contains one subfolder per source dataset, where the name of the subfolder matches the `{source}` in the attributes csv file. Each of these subfolders contains a shapefile with the catchment boundaries of each basin within that source dataset. These are the shapefiles that were used to derive the catchment attributes and ERA5-Land time series data. Each polygon in the shapefiles has on field, `gauge_id`, which contains the unique basin identifier (as defined in Caravan) in the form of `{source}_{id}`. Additionally, this folder contains one  shapefile called `combined.shp` that contains all basin polygons of Caravan in one layer.

- `code/`: This folder contains all scripts and Jupyter notebooks that were used to derive the data set. These scripts can be used to extend the data set to any new catchment in the world. Instructions are included in the `README.md` file contained in this folder. The folder is a static copy of [this GitHub repository](https://github.com/kratzert/Caravan) and contains the code from the time of this publication.

- `licenses/`: This folder contains a `README.md` file and one folder per source dataset. The `README.md` contains general information about the license of the Caravan dataset, ERA5-Land data, as well as data derived from HydroATLAS, while each subfolders contains information about the licenses of the source datasets. 

## Technical details

### Units of time series data

| Variable name                 | Description                               | Aggregation          | Unit  |
|-------------------------------|-------------------------------------------|----------------------|-------|
| snow_depth_water_equivalent   | Snow-Water Equivalent                     | Daily min, max, mean | mm    |
| surface_net_solar_radiation   | Surface net solar radiation               | Daily min, max, mean | W/m2  |
| surface_net_thermal_radiation | Surface net thermal radiation             | Daily min, max, mean | W/m2  |
| surface_pressure              | Surface pressure                          | Daily min, max, mean | kPa   |
| temperature_2m                | 2m air temperature                        | Daily min, max, mean | °C    |
| u_component_of_wind_10m       | U-component of wind at 10m                | Daily min, max, mean | m/s   |
| v_component_of_wind_10m       | V-component of wind at 10m                | Daily min, max, mean | m/s   |
| volumetric_soil_water_layer_1 | volumetric soil water layer 1 (0-7cm)     | Daily min, max, mean | m3/m3 |
| volumetric_soil_water_layer_2 | volumetric soil water layer 2 (7-28cm)    | Daily min, max, mean | m3/m3 |
| volumetric_soil_water_layer_3 | volumetric soil water layer 3 (28-100cm)  | Daily min, max, mean | m3/m3 |
| volumetric_soil_water_layer_4 | volumetric soil water layer 4 (100-289cm) | Daily min, max, mean | m3/m3 |
| total_precipitation           | Total precipitation                       | Daily sum            | mm    |
| potential_evaporation         | Potential Evapotranspiration              | Daily sum            | mm    |
| streamflow                    | Observed streamflow                       | Daily min, max, mean | mm/d  |

### Catchment attributes

Refer to the [Caravan paper](https://www.nature.com/articles/s41597-023-01975-w) for a detailed list of all static catchment attributes. Generally, there are two different sets of catchment attributes that are shared in two different files:

1. `attributes_hydroatlas_{source}.csv`: Attributes derived from the HydroATLAS dataset. Refer to the "BasinATLAS Catalogue" of HydroATLAS (see [here](https://www.hydrosheds.org/hydroatlas)) for an explanation of the features.
2. `attributes_caravan_{source}.csv`: Attributes (climate indices) derived from ERA5-Land timeseries for the Caravan dataset. See Table 4 in the [Caravan paper](https://www.nature.com/articles/s41597-023-01975-w) for details.
3. `attributes_other_{source}.csv`: Metadata information, such as station name, gauge latitude and logitude, country and area. See Table 5 in the [Caravan paper](https://www.nature.com/articles/s41597-023-01975-w) for details.

### Time zones
All data in Caravan are in the local time of the corresponding catchment.
We ignore any possible Daylight Saving Time, i.e., data for a given location is always in non-DST time, regardless of the date.

## Extend Caravan

To extend Caravan to new catchments, see the guides linked in the [GitHub repository](https://github.com/kratzert/Caravan).

## License

See `licenses/README.md` for details on the license of Caravan as well as all source datasets.

## How to cite

The Caravan dataset is currently under revision. If you already use Caravan in your research papers, please cite the dataset using the following reference.

```bib
@article{kratzert2023caravan,
  title={Caravan-A global community dataset for large-sample hydrology},
  author={Kratzert, Frederik and Nearing, Grey and Addor, Nans and Erickson, Tyler and Gauch, Martin and Gilon, Oren and Gudmundsson, Lukas and Hassidim, Avinatan and Klotz, Daniel and Nevo, Sella and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={61},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
