// Dataset: ISRIC/SoilGrids250m/v2_0
// Note: SoilGrids here is long-term baseline static information, not live time-series.

var soil = ee.Image('ISRIC/SoilGrids250m/v2_0/wv0010').select('val_0_5cm_Q0_5');

print('SoilGrids image', soil);

Map.setCenter(0, 20, 2);
Map.addLayer(
  soil,
  {min: 0.05, max: 0.6, palette: ['440154', '3b528b', '21918c', '5ec962', 'fde725']},
  'SoilGrids volumetric water content'
);
