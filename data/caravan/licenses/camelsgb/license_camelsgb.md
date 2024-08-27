# General information

The information contained in this file applies to:

- Streamflow time series under `timeseries/csv/camelsgb` and `timeseries/netcdf/camelsgb`
- Catchment boundaries under `shapefiles/camelsgb/`
- Gauge latitude and longitude and gauge name values in `attributes/camelsgb/attributes_other_camelsgb.csv`

This data was originally published as part of the CAMELS-GB dataset by Coxon et al. (2020) and is published under the terms of the [Open Government Licence v3](http://www.nationalarchives.gov.uk/doc/open-government-licence). Caravan therefore contains data supplied by Natural Environment Research Council.

**Note**: We do not include the `gauge_lat` and `gauge_lon` values that are included in the CAMELS-GB dataset, because they are rounded to two decimal digits, which is not sufficiently accurate. Instead, we use the provided `gauge_easting` and `gauge_northing` and compute lat/lon with the following function:

```python
from pyproj import Proj, transform

v84 = Proj(proj="latlong",towgs84="0,0,0",ellps="WGS84")
v36 = Proj(proj="latlong", k=0.9996012717, ellps="airy",
        towgs84="446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894")
vgrid = Proj(init="world:bng")


def easting_northing_to_lat_long(easting: int, northing: int):
    """Returns Latitude/Longitude (WGS84) derived from Easthing/Northing in UK."""
    vlon36, vlat36 = vgrid(easting, northing, inverse=True)
    return transform(v36, v84, vlon36, vlat36)
```

## References

Paper

```
Coxon, G., Addor, N., Bloomfield, J. P., Freer, J., Fry, M., Hannaford, J., Howden, N. J. K., Lane, R., Lewis, M., Robinson, E. L., Wagener, T., and Woods, R.: CAMELS-GB: hydrometeorological time series and landscape attributes for 671 catchments in Great Britain, Earth Syst. Sci. Data, 12, 2459–2483, https://doi.org/10.5194/essd-12-2459-2020, 2020.
```

Dataset
```
Coxon, G.; Addor, N.; Bloomfield, J.P.; Freer, J.; Fry, M.; Hannaford, J.; Howden, N.J.K.; Lane, R.; Lewis, M.; Robinson, E.L.; Wagener, T.; Woods, R. (2020). Catchment attributes and hydro-meteorological timeseries for 671 catchments across Great Britain (CAMELS-GB). NERC Environmental Information Data Centre. https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9
```