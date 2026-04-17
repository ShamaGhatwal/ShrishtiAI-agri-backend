// Datasets:
// - FAO/WAPOR/3/L1_AETI_D
// - FAO/WAPOR/3/L1_RET_D
// - FAO/WAPOR/3/L1_T_D

var start = '2024-01-01';
var end = '2024-12-31';

var aeti = ee.ImageCollection('FAO/WAPOR/3/L1_AETI_D')
  .filterDate(start, end)
  .mean();

var ret = ee.ImageCollection('FAO/WAPOR/3/L1_RET_D')
  .filterDate(start, end)
  .mean();

var t = ee.ImageCollection('FAO/WAPOR/3/L1_T_D')
  .filterDate(start, end)
  .mean();

var etRatio = aeti.select('L1-AETI-D').divide(ret.select('L1-RET-D')).rename('ETa_ETo_ratio');

print('AETI sample', aeti);
print('RET sample', ret);
print('Transpiration sample', t);

Map.setCenter(20, 15, 2);
Map.addLayer(etRatio, {min: 0, max: 1.5, palette: ['8b0000', 'ffa500', 'ffff00', '00ff00']}, 'ETa/ETo ratio');
Map.addLayer(t.select('L1-T-D'), {min: 0, max: 8, palette: ['f7fbff', '6baed6', '08306b']}, 'Transpiration (mm/day)');
