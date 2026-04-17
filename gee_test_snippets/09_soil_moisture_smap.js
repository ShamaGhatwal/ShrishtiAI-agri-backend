// Datasets:
// - NASA/SMAP/SPL3SMP_E/006 (surface)
// - NASA/SMAP/SPL4SMGP/008 (surface + root zone)

var l3 = ee.ImageCollection('NASA/SMAP/SPL3SMP_E/006')
  .filterDate('2024-01-01', '2024-12-31')
  .select('soil_moisture_am')
  .mean();

var l4 = ee.ImageCollection('NASA/SMAP/SPL4SMGP/008')
  .filterDate('2024-01-01', '2024-12-31')
  .mean();

print('SMAP L3 mean', l3);
print('SMAP L4 mean', l4);
print('SMAP L4 bands', l4.bandNames());

Map.setCenter(0, 20, 2);
Map.addLayer(l3, {min: 0, max: 0.6, palette: ['f46d43', 'fdae61', 'abd9e9', '2c7bb6']}, 'SMAP L3 surface moisture');
Map.addLayer(l4.select('sm_rootzone'), {min: 0, max: 0.6, palette: ['8c510a', 'd8b365', '5ab4ac', '01665e']}, 'SMAP L4 rootzone moisture');
