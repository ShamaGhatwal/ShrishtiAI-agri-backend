// Dataset:
// projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km/v1_2_2001_2024

var img = ee.Image('projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km/v1_2_2001_2024');

print('Forest drivers image', img);
print('Band names', img.bandNames());

Map.setCenter(0, 20, 2);
Map.addLayer(
  img.select(0),
  {
    min: 1,
    max: 7,
    palette: ['fdae61', 'd7191c', 'abdda4', '2b83ba', 'f46d43', '8073ac', '999999']
  },
  'Dominant forest loss driver (band 0)'
);
