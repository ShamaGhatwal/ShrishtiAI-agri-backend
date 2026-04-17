// Dataset: UTOKYO/WTLAB/KBDI/v1
var kbdi = ee.ImageCollection('UTOKYO/WTLAB/KBDI/v1')
  .filterDate('2024-01-01', '2024-12-31')
  .sort('system:time_start', false);

var latest = ee.Image(kbdi.first());

print('KBDI latest image', latest);
print('KBDI bands', latest.bandNames());

Map.setCenter(0, 20, 2);
Map.addLayer(
  latest.select('KBDI'),
  {min: 0, max: 800, palette: ['313695', '74add1', 'fee090', 'f46d43', 'a50026']},
  'KBDI'
);
