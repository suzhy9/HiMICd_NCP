# %%[1 Obtain data - by GEE]
# Note:this section code includes two parts:
# (1) obtaining daily multi-band images,
# (2) extracting daily gridded datasets using meteorological stations.

# =========================(1) obtaining daily multi-band images===============================
//Function → obtain daily LST collection of specific year
// Load gap-filled daily LST collections (1 km)
var gf_day = ee.ImageCollection("projects/sat-io/open-datasets/gap-filled-lst/gf_day_1km");
var gf_night = ee.ImageCollection("projects/sat-io/open-datasets/gap-filled-lst/gf_night_1km");

// Define study region
var aoi = ee.FeatureCollection('projects/ee-szying168/assets/HTP_area_shp');
var geometry = aoi.geometry().bounds();

// -------------------------
// Parameters
// -------------------------
var year = 2013;            // Year to export
var folder = 'LST2013';     // Google Drive folder
var scale = 1000;           // Spatial resolution in meters

// -------------------------
// Compute start index for the year
// -------------------------
var startIndex = (year - 2003) * 365;  // assuming 365 days per year
var nDays = 365;

// -------------------------
// Convert ImageCollections to lists and slice the images for the selected year
// -------------------------
var dayList = gf_day.toList(startIndex + nDays).slice(startIndex, startIndex + nDays);
var nightList = gf_night.toList(startIndex + nDays).slice(startIndex, startIndex + nDays);

// -------------------------
// Loop through each day and export
// -------------------------
for (var i = 0; i < nDays; i++) {
  var dayImg = ee.Image(dayList.get(i));
  var nightImg = ee.Image(nightList.get(i));

  // Compute daily mean of day/night LST and apply scaling factor
  var imgMean = ee.ImageCollection.fromImages([dayImg, nightImg])
                    .mean()
                    .multiply(0.1)
                    .clip(geometry);

  // Get short ID from system:id
  var fullId = dayImg.get('system:id');
  var shortId = ee.String(fullId).split('/').get(-1);
  imgMean = imgMean.set('system:index', shortId);

  // Extract date string from image ID for file naming
  var dateStr = ee.String(fullId).split('_').get(-1).getInfo();

  // Export image to Google Drive
  Export.image.toDrive({
    image: imgMean,
    description: 'DailyLST_' + year + '_' + dateStr,
    folder: folder,
    fileNamePrefix: 'LST_' + year + '_' + dateStr,
    region: geometry,
    scale: scale,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });

  // Print progress to console
  print('Exporting day', i+1, 'of', nDays, 'for year', year, 'Date:', dateStr);
}

# =============================================================================================
//Function → obtain yearly POP collection of specific year
// -------------------------
// Define study region
// -------------------------
var aoi = ee.FeatureCollection('projects/ee-szying168/assets/HTP_area_shp');
var geometry = aoi.geometry().bounds();

// -------------------------
// Parameters
// -------------------------
var year = 2020;              // Year of population data
var folder = 'Pop2020';  // Google Drive folder
var scale = 1000;             // Output spatial resolution in meters

// -------------------------
// Load WorldPop population dataset
// -------------------------
var pop_dataset = ee.ImageCollection("WorldPop/GP/100m/pop")
                      .select('population')
                      .filterDate(year + '-01-01', year + '-12-31');

// WorldPop generally provides one image per year; use median to get a single image
var pop_img = pop_dataset.median()
                    .setDefaultProjection('EPSG:4326', null, 100) // original 100m
                    .reduceResolution({
                      reducer: ee.Reducer.sum(),
                      maxPixels: 1024
                    })
                    .reproject({
                      crs: 'EPSG:4326',
                      scale: scale
                    })
                    .clip(geometry)
                    .rename('pop')
                    .unmask(-9999); // fill missing values with -9999

// -------------------------
// Export population image to Google Drive
// -------------------------
Export.image.toDrive({
  image: pop_img,
  description: 'Pop_' + year,
  folder: folder,
  fileNamePrefix: 'Population_' + year,
  region: geometry,
  scale: scale,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Print message
print('Population data for', year, 'export initiated.');

# =============================================================================================
//Function → obtain daily Precipitable water vapor collection of specific year
// -------------------------
// Define study area
// -------------------------
var aoi = ee.FeatureCollection('projects/ee-szying168/assets/HTP_area_shp');
var geometry = aoi.geometry().bounds();

// -------------------------
// Parameters
// -------------------------
var startDate = '2020-11-01';   // Start date (YYYY-MM-DD)
var endDate = '2021-01-01';     // End date (YYYY-MM-DD)
var folder = 'WV2020';          // Google Drive folder
var scale = 1000;               // Export spatial resolution in meters
var area = geometry;            // Export region

// -------------------------
// Load MODIS water vapor dataset
// -------------------------
var wv_dataset = ee.ImageCollection('MODIS/006/MCD19A2_GRANULES')
                    .select('Column_WV')
                    .filterDate(startDate, endDate);

// -------------------------
// Generate sequence of days
// -------------------------
var start = ee.Date(startDate);
var end = ee.Date(endDate);
var days = ee.List.sequence(0, end.difference(start, 'day').subtract(1));

// -------------------------
// Loop through each day and export
// -------------------------
days.getInfo().forEach(function(d) {
  var date = start.advance(d, 'day');
  var dateStr = date.format('yyyyMMdd').getInfo();

  // Filter images for the specific day
  var daily = wv_dataset.filterDate(date, date.advance(1, 'day'));

  // If image exists, compute mean; otherwise fill with -9999
  var img = ee.Algorithms.If(
    daily.size(),
    daily.mean()
        .multiply(0.001)       // Apply scale factor
        .clip(area)
        .rename('wv')
        .unmask(-9999),        // Fill missing values with -9999
    ee.Image(-9999).rename('wv').clip(area)
  );

  // Export to Google Drive
  Export.image.toDrive({
    image: ee.Image(img),
    description: 'WV_' + dateStr,
    folder: folder,
    fileNamePrefix: 'WaterVapor_' + dateStr,
    region: area,
    scale: scale,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });

  // Print progress
  print('Exporting water vapor for date:', dateStr);
});

# =============================================================================================
//Function → obtain yearly Landcover collection of each year
// -----------------------------
// Parameters (Easy to edit)
// -----------------------------
var startYear = 2003;                    // Start year
var endYear = 2021;                      // End year
var datasetId = 'MODIS/061/MCD12Q1';     // MODIS Land Cover dataset
var bandName = 'LC_Type1';               // Land cover band to export
var folderName = 'MODIS_LandCover';      // Google Drive export folder
var exportScale = 1000;                   // Spatial resolution (meters)
var projection = 'EPSG:4326';            // CRS projection
var regionAsset = 'projects/ee-szying168/assets/HTP_area_shp'; // AOI shapefile

// -----------------------------
// Load region and dataset
// -----------------------------
var aoi = ee.FeatureCollection(regionAsset);
var bounds = aoi.geometry().bounds();

// Load the MODIS Land Cover dataset
var dataset = ee.ImageCollection(datasetId).select(bandName);

// -----------------------------
// Loop through each year
// -----------------------------
for (var year = startYear; year <= endYear; year++) {
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate = ee.Date.fromYMD(year + 1, 1, 1);

  // Filter by year and get the first image
  var landCover = dataset.filterDate(startDate, endDate).first();

  // Ensure valid image
  landCover = ee.Image(landCover);
  if (landCover) {
    var clipped = landCover.clip(bounds);

    // Optional: add to map for visualization
    Map.addLayer(clipped, igbpVis, 'Land Cover ' + year);

    // Export to Google Drive
    Export.image.toDrive({
      image: clipped,
      description: 'IGBP_LandCover_' + year,
      folder: folderName,
      fileNamePrefix: 'IGBP_LandCover_' + year,
      region: bounds,
      scale: exportScale,
      crs: projection,
      maxPixels: 1e13
    });

    print('Export task created for year:', year);
  } else {
    print('No land cover data found for year:', year);
  }
}

# =============================================================================================
// Flexible DEM + Slope + DOY Export Script
// Generates yearly composites (2003-2020)
// -----------------------------
// Parameters (easy to edit)
// -----------------------------
var startYear = 2003;
var endYear = 2020;
var folder = 'DEM_Slope_DOY';
var scale = 1000;
var aoi = ee.FeatureCollection(regionAsset);
var region = aoi.geometry().bounds();

// -----------------------------
// Base DEM and slope (constant each year)
// -----------------------------
var dem = ee.Image("MERIT/DEM/v1_0_3")
              .select("dem")
              .reproject('EPSG:4326', null, scale)
              .clip(region)
              .unmask(-9999)
              .rename('dem');

var slope = ee.Terrain.slope(dem)
              .reproject('EPSG:4326', null, scale)
              .clip(region)
              .unmask(-9999)
              .rename('slope');

// -----------------------------
// Loop through years
// -----------------------------
for (var year = startYear; year <= endYear; year++) {
  var t1 = ee.Date.fromYMD(year, 1, 1);
  var doy = t1.getRelative("day", "year");  // DOY for Jan 1

  var dayOfYear = ee.Image.constant(doy)
                    .setDefaultProjection('EPSG:4326', null, scale)
                    .clip(region)
                    .unmask(-9999)
                    .rename('doy')
                    .toFloat();

  // Merge DEM, slope, and DOY into one image
  var composite = dem.addBands(slope).addBands(dayOfYear);

  // Add to map (optional)
  Map.addLayer(composite, {}, 'DEM_Slope_DOY_' + year);

  // Export to Google Drive
  Export.image.toDrive({
    image: composite,
    description: 'DEM_Slope_DOY_' + year,
    folder: folder,
    fileNamePrefix: 'DEM_Slope_DOY_' + year,
    region: region,
    scale: scale,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });

  print('Export task created for year:', year);
}

# =============================================================================================
// ERA5-Land Humidity Variables

// Define study period (edit freely)
var START_DATE = '2013-01-01';       // Start date
var END_DATE   = '2014-01-01';       // End date

// Define study area (example: China extent)
var aoi = ee.FeatureCollection(regionAsset);
var REGION = aoi.geometry().bounds();

// Export settings
var SCALE = 11132;                     // Export resolution (m)
var FOLDER = 'HTP_humidity_exports'; // Drive folder name
var CRS = 'EPSG:4326';                 // Coordinate reference system

// -----------------------------
// LOAD ERA5-LAND DATASET
// -----------------------------
var dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
  .filterDate(START_DATE, END_DATE)
  .filterBounds(REGION)
  .select(['temperature_2m', 'dewpoint_temperature_2m']);  // Only required for calculation

print('Filtered ERA5-Land dataset:', dataset);

// -----------------------------
// DEFINE PHYSICAL CALCULATION FUNCTIONS
// -----------------------------

// (1) Saturation vapor pressure (hPa) using Magnus-Tetens formula
function saturationVaporPressure(T_C) {
  return ee.Image(6.112).multiply(
    T_C.expression('exp((17.67 * T) / (T + 243.5))', {'T': T_C})
  );
}

// (2) Relative Humidity (RH, %)
function calculateRH(T_K, Td_K) {
  var T_C  = T_K.subtract(273.15);   // Convert Kelvin → Celsius
  var Td_C = Td_K.subtract(273.15);
  var es   = saturationVaporPressure(T_C);
  var esd  = saturationVaporPressure(Td_C);
  return esd.divide(es).multiply(100).rename('rh');
}

// (3) Actual vapor pressure (AVP, hPa)
function actualVaporPressure(Td_C) {
  return saturationVaporPressure(Td_C).rename('avp');
}

// (4) Vapor pressure deficit (VPD, hPa)
function vaporPressureDeficit(T_C, Td_C) {
  var esT = saturationVaporPressure(T_C);
  var avp = saturationVaporPressure(Td_C);
  return esT.subtract(avp).rename('vpd');
}

// (5) Mixing ratio (MR, g/kg)
function mixingRatio(avp) {
  var P = ee.Image(1013.25);  // Assume standard atmospheric pressure
  return avp.multiply(0.622).divide(P.subtract(avp)).rename('mr');
}

// (6) Specific humidity (SH)
function specificHumidity(mr) {
  return mr.divide(mr.add(1)).rename('sh');
}

// -----------------------------
// COMPUTE DAILY HUMIDITY VARIABLES
// -----------------------------
var humidityCol = dataset.map(function(image) {
  var T_K  = image.select('temperature_2m');
  var Td_K = image.select('dewpoint_temperature_2m');
  var T_C  = T_K.subtract(273.15);
  var Td_C = Td_K.subtract(273.15);

  // Compute each variable
  var rh  = calculateRH(T_K, Td_K);
  var avp = actualVaporPressure(Td_C);
  var vpd = vaporPressureDeficit(T_C, Td_C);
  var mr  = mixingRatio(avp);
  var sh  = specificHumidity(mr);

  // Combine all into one image
  var out = ee.Image.cat([rh, avp, vpd, Td_C.rename('dpt'), mr, sh])
    .copyProperties(image, ["system:time_start"]);

  return out;
});

print('Processed humidity image collection:', humidityCol);

// -----------------------------
// FLEXIBLE EXPORT FUNCTION
// -----------------------------
function exportImageCollection(imgCol, region, prefix) {
  var list = imgCol.toList(imgCol.size());
  var count = list.size().getInfo();

  print('Exporting', count, 'images...');

  for (var i = 0; i < count; i++) {
    var image = ee.Image(list.get(i));
    var date = ee.Date(image.get('system:time_start')).format('yyyyMMdd').getInfo();

    Export.image.toDrive({
      image: image.select(['rh', 'avp', 'vpd', 'dpt', 'mr', 'sh']),  // Export only humidity bands
      description: prefix + '_humidity_' + date,
      fileNamePrefix: prefix + '_humidity_' + date,
      folder: FOLDER,
      region: region,
      scale: SCALE,
      crs: CRS,
      maxPixels: 1e13
    });
  }
}

// -----------------------------
// RUN EXPORT
// -----------------------------
exportImageCollection(humidityCol, REGION, 'ERA5L');


# =================(2) extracting daily gridded datasets using meteorological stations=================
// ============================================================
// POINT EXTRACTION: LST + POP + DEM + SLOPE + MODIS WV + LC + DOY + ERA5 HUMIDITY
// ============================================================

// -----------------------------
// USER PARAMETERS (edit these)
// -----------------------------
var year = 2020;
var stationsAsset = "users/szying168/HiTIC-NCP/point"; // station points
var regionAsset = 'projects/ee-szying168/assets/HTP_area_shp'; // AOI shapefile
var scale = 1000;                     // sampling scale (m)
var folder = 'Station_Extractions';   // Drive folder
var filePrefix = 'station_allvars_' + year;

// -----------------------------
// Derived dates and region
// -----------------------------
var startDate = ee.Date.fromYMD(year, 1, 1);
var endDate   = ee.Date.fromYMD(year + 1, 1, 1);
var stations = ee.FeatureCollection(stationsAsset);
var region = regionAsset ? ee.FeatureCollection(regionAsset).geometry().bounds() : stations.geometry().bounds();

// sentinel for missing data
var FILL = -9999;

// -----------------------------
// Load / prepare static datasets
// -----------------------------

// Population
var pop = ee.ImageCollection("WorldPop/GP/100m/pop")
           .filterDate(startDate, endDate)
           .median()
           .rename('pop')
           .reproject('EPSG:4326', null, 1000)
           .clip(region);

// DEM and slope (MERIT)
var dem = ee.Image("MERIT/DEM/v1_0_3").select('dem').rename('elevation').clip(region);
var slope = ee.Terrain.slope(dem).rename('slope').clip(region);

// Land cover (MODIS MCD12Q1)
var lc = ee.ImageCollection('MODIS/061/MCD12Q1')
           .filterDate(startDate, endDate)
           .select('LC_Type1')
           .first();
lc = ee.Image(ee.Algorithms.If(lc, lc.rename('landcover').clip(region), ee.Image.constant(FILL).rename('landcover').clip(region)));

// -----------------------------
// Collections used per-day
// -----------------------------
var gf_day = ee.ImageCollection("projects/sat-io/open-datasets/gap-filled-lst/gf_day_1km")
               .filterDate(startDate, endDate).filterBounds(region);
var gf_night = ee.ImageCollection("projects/sat-io/open-datasets/gap-filled-lst/gf_night_1km")
                 .filterDate(startDate, endDate).filterBounds(region);

// MODIS column water vapour (MCD19A2_GRANULES)
var wv_collection = ee.ImageCollection("MODIS/006/MCD19A2_GRANULES")
                        .select('Column_WV')
                        .filterDate(startDate, endDate)
                        .filterBounds(region);

// ERA5-Land daily aggregation (for humidity variables)
var era5_daily = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                    .filterDate(startDate, endDate)
                    .filterBounds(region)
                    .select(['temperature_2m', 'dewpoint_temperature_2m']);

// -----------------------------
// Helper: safe single-band retrieval (returns image or fill)
// -----------------------------
function getDailySingleBand(collection, date, scaleFactor, outName, fillValue){
  var filtered = collection.filterDate(date, date.advance(1, 'day'));
  return ee.Image(ee.Algorithms.If(
    filtered.size(),
    filtered.mean().multiply(scaleFactor).rename(outName).clip(region),
    ee.Image.constant(fillValue).rename(outName).clip(region)
  ));
}

// -----------------------------
// Helper: safe multi-band LST (day/night mean) -> LST band (units after multiply)
// returns image with band 'LST'
// -----------------------------
function getDailyLST(date){
  var dayImgCol = gf_day.filterDate(date, date.advance(1, 'day'));
  var nightImgCol = gf_night.filterDate(date, date.advance(1, 'day'));
  var hasDay = dayImgCol.size();
  var hasNight = nightImgCol.size();

  // if both absent return fill image
  var result = ee.Image(ee.Algorithms.If(
    ee.Algorithms.If(hasDay, 1, 0).add(ee.Algorithms.If(hasNight, 1, 0)).eq(0),
    ee.Image.constant(FILL).rename('LST').clip(region),
    ee.Image( // else compute mean of available images
      ee.Algorithms.If(
        // both present -> mean
        hasDay.and(hasNight),
        ee.ImageCollection.fromImages([dayImgCol.first(), nightImgCol.first()]).mean().multiply(0.1).rename('LST').clip(region),
        // only one present -> use that one
        ee.Algorithms.If(hasDay, ee.Image(dayImgCol.first()).multiply(0.1).rename('LST').clip(region), ee.Image(nightImgCol.first()).multiply(0.1).rename('LST').clip(region))
      )
    )
  ));
  return ee.Image(result);
}

// -----------------------------
// Helper: compute ERA5 humidity variables for a given date,
// returns image with bands ['rh','avp','vpd','dpt','mr','sh'] or fill bands
// -----------------------------
function calculateERA5HumidityForDate(date){
  var filtered = era5_daily.filterDate(date, date.advance(1, 'day'));
  return ee.Image(ee.Algorithms.If(filtered.size(),
    // if available, compute from mean (or first)
    (function(){
      var img = ee.Image(filtered.mean());
      var T_K = img.select('temperature_2m');
      var Td_K = img.select('dewpoint_temperature_2m');
      var T_C = T_K.subtract(273.15);
      var Td_C = Td_K.subtract(273.15);

      // Magnus-Tetens saturation vapor pressure (hPa)
      function sat_vp(TC){
        return ee.Image(6.112).multiply(TC.expression('exp((17.67 * T) / (T + 243.5))', {'T': TC}));
      }

      var es = sat_vp(T_C);
      var esd = sat_vp(Td_C);
      var rh = esd.divide(es).multiply(100).rename('rh');
      var avp = esd.rename('avp');
      var vpd = es.subtract(esd).rename('vpd');
      var P = ee.Image(1013.25);
      var mr = avp.multiply(0.622).divide(P.subtract(avp)).multiply(1000).rename('mr'); // convert to g/kg (optional *1000)
      var sh = mr.divide(mr.add(1)).rename('sh'); // note: mr dimensionless here if not scaled; keep consistent

      // return concatenated image including dewpoint (°C)
      return ee.Image.cat([rh, avp, vpd, Td_C.rename('dpt'), mr, sh]).clip(region);
    })(),
    // else return constant image with same band names filled with FILL
    ee.Image.constant([FILL, FILL, FILL, FILL, FILL, FILL]).rename(['rh','avp','vpd','dpt','mr','sh']).clip(region)
  ));
}

// -----------------------------
// Create list of day offsets and iterate (client-side evaluate used to drive the loop)
// -----------------------------
var nDays = endDate.difference(startDate, 'day').toInt();
var dayOffsets = ee.List.sequence(0, nDays.subtract(1));

var allSamples = ee.FeatureCollection([]);

// evaluate the list client-side to loop (keeps export tasks discrete)
dayOffsets.getInfo().forEach(function(offset){
  var current = startDate.advance(offset, 'day');
  var dateStr = current.format('YYYY-MM-dd').getInfo();
  var doy = current.getRelative('day', 'year').getInfo() + 1; // 1-based DOY

  // LST (K * 0.1 -> multiply by 0.1 gives original units; adjust if needed)
  var lstImg = getDailyLST(current);

  // MODIS WV (Column_WV) - apply scale factor 0.001 -> units (e.g., mm or cm depending on product)
  var wvImg = getDailySingleBand(wv_collection, current, 0.001, 'WV', FILL);

  // ERA5 humidity vars for the day
  var era5Img = calculateERA5HumidityForDate(current);

  // Static layers (reproject/populate to region if necessary)
  var popImg = pop.rename('POP').reproject('EPSG:4326', null, 1000).clip(region);
  var demImg = dem.rename('ELEV').reproject('EPSG:4326', null, 1000).clip(region);
  var slopeImg = slope.rename('SLOPE').reproject('EPSG:4326', null, 1000).clip(region);
  var lcImg = lc.rename('LC').reproject('EPSG:4326', null, 1000).clip(region);

  // DOY image (single band)
  var doyImg = ee.Image.constant(doy).rename('DOY').reproject('EPSG:4326', null, 1000).clip(region);

  // Merge into one image with consistent band names
  var combined = ee.Image.cat([
    lstImg.select('LST'),
    popImg.select('pop').rename('POP'),
    demImg.select('elevation').rename('ELEV'),
    slopeImg.select('SLOPE'),
    lcImg.select('LC'),
    wvImg.select('WV'),
    era5Img.select(['rh','avp','vpd','dpt','mr','sh']),
    doyImg.select('DOY')
  ]).set('date', dateStr);

  // Sample at stations
  var samples = combined.sampleRegions({
    collection: stations,
    properties: ['station'], // carry station ID/name if exists
    scale: scale,
    geometries: true
  }).map(function(f){
    return f.set('date', dateStr);
  });

  // accumulate
  allSamples = allSamples.merge(samples);

  print('Prepared samples for', dateStr);
});

// Final export of collected points (single CSV for the year)
Export.table.toDrive({
  collection: allSamples,
  description: filePrefix,
  folder: folder,
  fileNamePrefix: filePrefix,
  fileFormat: 'CSV'
});

print('Export task created:', filePrefix);

# ====================================================================================================
# %%[2 Data cleaning]
###------ load data ------
data2003_2020_path = r"E:\high-resolution atmospheric moisture\Data\1 Allcombined_cleanedData\CombinedALLData2003_2020.csv"
data = pd.read_csv(data2003_2020_path)
unname_col = data.filter(regex = "Unname")
data = data.drop(unname_col, axis=1)
columnsList = data.columns[1:]
print(data.shape)

###------ cleaning ------
for i in columnsList:
  data_cln = data.loc[~data[i].isin([-9999.0])]
  print("-9999: {} left-{}".format(i,data_cln.shape))

for m in columnsList:
    data_cln = data_cln.dropna(axis='index', how='any',subset=[m])
print("Nan: {} left-{}".format(m,data_cln.shape))

###------ export ------
out_path = r"E:\high-resolution atmospheric moisture\Data\1 Allcombined_cleanedData\Cleaned_ALLData2003_2020.csv"
data_cln.to_csv(out_path, encoding='utf-8')


# ====================================================================================================
# %%[3 Splitting data into training sets and validation sets]
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

###------ load data ------
year = 2003

cleaningData_path = r"E:\high-resolution atmospheric moisture\Data\1 Allcombined_cleanedData\Cleaned_ALLData2003_2020.csv"
cleaningData = pd.read_csv(cleaningData_path)
ss = cleaningData.filter(regex = "Unname")
cleaningData = cleaningData.drop(ss, axis = 1)
del ss,cleaningData_path

###------ split training/validation ------
percent = 0.2
id_list = np.unique(cleaningData['id'])
TrainData = pd.DataFrame()
TestData = pd.DataFrame()
num=0

for i in id_list:
  id_data = cleaningData.loc[cleaningData["id"].isin([i])] #split by individual stations
  id_traindata, id_testdata = train_test_split(id_data, test_size=percent, random_state=42)
  TrainData = pd.concat([TrainData,id_traindata],axis=0,ignore_index=False)
  TestData = pd.concat([TestData,id_testdata],axis=0,ignore_index=False)
  print("No.{} has done, which id is {}".format(num,i))
  num+=1

print("original:",cleaningData.shape,
      "training:",TrainData.shape,
      "validation:",TestData.shape)
TrainData.to_csv(r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Train_all.csv".format(year), encoding='utf-8')
TestData.to_csv(r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Test_all.csv".format(year), encoding='utf-8')
del percent, id_list, num, i, id_data, id_traindata, id_testdata

