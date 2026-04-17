// Dataset: NASA/GPM_L3/IMERG_V07
var imerg = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
  .filterDate('2024-06-01', '2024-06-30');

var precip = imerg.select('precipitation').mean();

print('IMERG first image', imerg.first());
print('IMERG count', imerg.size());

Map.setCenter(0, 20, 2);
Map.addLayer(
  precip,
  {min: 0, max: 10, palette: ['f7fbff', 'c6dbef', '6baed6', '2171b5', '08306b']},
  'IMERG precipitation'
);
