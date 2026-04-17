// Datasets:
// - GOOGLE/DYNAMICWORLD/V1
// - ESA/WorldCover/v200

var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate('2023-01-01', '2023-12-31');

var cropProb = dw.select('crops').mean();
var worldCover = ee.ImageCollection('ESA/WorldCover/v200').first();

print('Dynamic World count', dw.size());
print('WorldCover image', worldCover);

Map.setCenter(0, 20, 2);
Map.addLayer(cropProb, {min: 0, max: 1, palette: ['ffffff', 'ffffb2', 'fecc5c', 'e31a1c']}, 'DW crops probability');
Map.addLayer(worldCover.select('Map'), {min: 10, max: 100, palette: ['006400', 'ffbb22', 'ffff4c', 'f096ff', 'fa0000', 'b4b4b4', 'f0f0f0', '0064c8', '0096a0', '00cf75', 'fae6a0']}, 'ESA WorldCover 2021');
