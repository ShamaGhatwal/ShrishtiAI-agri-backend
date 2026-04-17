// Dataset: CSIC/SPEI/2_10
var spei = ee.ImageCollection('CSIC/SPEI/2_10')
  .filterDate('2022-01-01', '2023-01-01')
  .sort('system:time_start', false);

var latest = ee.Image(spei.first());

print('SPEI latest image', latest);
print('SPEI bands', latest.bandNames());

Map.setCenter(0, 20, 2);
Map.addLayer(
  latest.select('SPEI_12_month'),
  {min: -2.5, max: 2.5, palette: ['8b0000', 'f46d43', 'fee08b', 'd9ef8b', '66bd63', '1a9850']},
  'SPEI 12-month'
);
