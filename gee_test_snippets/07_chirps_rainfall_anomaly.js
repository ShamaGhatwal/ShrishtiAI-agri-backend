// Dataset: UCSB-CHG/CHIRPS/DAILY
var recentStart = '2024-06-01';
var recentEnd = '2024-08-31';

var baseStart = '2014-06-01';
var baseEnd = '2023-08-31';

var recent = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(recentStart, recentEnd)
  .select('precipitation')
  .sum();

var baseline = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(baseStart, baseEnd)
  .select('precipitation')
  .mean();

var anomaly = recent.subtract(baseline).rename('rain_anomaly_mm');

print('Recent total rainfall image', recent);
print('Baseline mean image', baseline);

Map.setCenter(0, 20, 2);
Map.addLayer(anomaly, {min: -200, max: 200, palette: ['8b0000', 'fdd49e', 'f7f7f7', '9ecae1', '08519c']}, 'CHIRPS anomaly');
