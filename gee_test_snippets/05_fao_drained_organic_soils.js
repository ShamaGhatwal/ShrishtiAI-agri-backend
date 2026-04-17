// Datasets:
// - FAO/GHG/1/DROSA_A (drained organic soil area)
// - FAO/GHG/1/DROSE_A (emissions)

var drosa = ee.ImageCollection('FAO/GHG/1/DROSA_A').sort('system:time_start', false).first();
var drose = ee.ImageCollection('FAO/GHG/1/DROSE_A').sort('system:time_start', false).first();

var drosaImg = ee.Image(drosa);
var droseImg = ee.Image(drose);

print('DROSA latest image', drosaImg);
print('DROSE latest image', droseImg);
print('DROSA bands', drosaImg.bandNames());
print('DROSE bands', droseImg.bandNames());

Map.setCenter(0, 20, 2);
Map.addLayer(drosaImg.select(0), {min: 0, max: 50, palette: ['f7fcf5', '74c476', '00441b']}, 'DROSA latest (band 0)');
Map.addLayer(droseImg.select(0), {min: 0, max: 50, palette: ['fff5f0', 'fb6a4a', '67000d']}, 'DROSE latest (band 0)');
