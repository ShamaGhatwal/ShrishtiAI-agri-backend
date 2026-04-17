// Dataset: USGS/GFSAD1000_V1
var gfsad = ee.Image('USGS/GFSAD1000_V1');
var lc = gfsad.select('landcover');

print('GFSAD image', gfsad);

Map.setCenter(0, 20, 2);
Map.addLayer(
  lc,
  {min: 0, max: 5, palette: ['000000', 'ff8c00', '8b4513', '00a650', '7fff00', 'ffff00']},
  'GFSAD landcover'
);
