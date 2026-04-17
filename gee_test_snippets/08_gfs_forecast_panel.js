// Dataset: NOAA/GFS0P25
var gfs = ee.ImageCollection('NOAA/GFS0P25')
  .filter(ee.Filter.gte('forecast_hours', 0))
  .filter(ee.Filter.lte('forecast_hours', 24))
  .sort('system:time_start', false);

var first = ee.Image(gfs.first());

print('GFS sample image', first);
print('GFS band names', first.bandNames());

Map.setCenter(0, 20, 2);
Map.addLayer(
  first.select('temperature_2m_above_ground'),
  {min: 10, max: 45, palette: ['313695', '74add1', 'fee090', 'f46d43', 'a50026']},
  '2m temp forecast (C)'
);
