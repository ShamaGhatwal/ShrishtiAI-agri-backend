// Dataset: ESA/WorldCereal/2021/MODELS/v100

var wcModels = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100');

print('WorldCereal MODELS count', wcModels.size());
print('First image', wcModels.first());

var first = ee.Image(wcModels.first());
Map.setCenter(0, 20, 2);
Map.addLayer(first.select('classification'), {min: 0, max: 100, palette: ['000000', '00ff00']}, 'WC classification');
Map.addLayer(first.select('confidence'), {min: 0, max: 100, palette: ['ff0000', 'ffff00', '00ff00']}, 'WC confidence');
