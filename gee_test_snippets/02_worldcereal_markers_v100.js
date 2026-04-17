// Dataset: ESA/WorldCereal/2021/MARKERS/v100

var wcMarkers = ee.ImageCollection('ESA/WorldCereal/2021/MARKERS/v100');

print('WorldCereal MARKERS count', wcMarkers.size());
print('First image', wcMarkers.first());

var first = ee.Image(wcMarkers.first());
Map.setCenter(0, 20, 2);
Map.addLayer(first.select('classification'), {min: 0, max: 100, palette: ['000000', '00bfff']}, 'Active cropland marker');
Map.addLayer(first.select('confidence'), {min: 0, max: 100, palette: ['8b0000', 'ffd700', '00ff7f']}, 'Marker confidence');
