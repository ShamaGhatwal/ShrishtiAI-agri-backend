// Dataset: ECMWF/ERA5_LAND/HOURLY
var era = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .filterDate('2024-05-01', '2024-05-31');

var t2m = era.select('temperature_2m').mean().subtract(273.15).rename('t2m_c');

print('ERA5-Land sample', era.first());
print('ERA5-Land count', era.size());

Map.setCenter(0, 20, 2);
Map.addLayer(t2m, {min: 15, max: 45, palette: ['313695', '74add1', 'fee08b', 'f46d43', 'a50026']}, 'ERA5-Land mean 2m temp (C)');
