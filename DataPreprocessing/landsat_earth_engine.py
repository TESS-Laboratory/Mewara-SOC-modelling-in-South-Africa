import ee

# Initialize the Earth Engine library.
ee.Initialize()

# Load the feature collection for South Africa
table = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
SA = table.filter(ee.Filter.eq("country_na", "South Africa"))

# Function to mask clouds, cloud shadows, and snow
def maskCloudAndShadows(image):
    qa = image.select('QA_PIXEL')
    
    # Bits 3, 4, and 5 are cloud, cloud shadow, and snow respectively.
    cloudsBitMask = (1 << 3)
    cloudShadowBitMask = (1 << 4)
    snowBitMask = (1 << 5)

    mask = qa.bitwiseAnd(cloudsBitMask).eq(0)\
             .And(qa.bitwiseAnd(cloudShadowBitMask).eq(0))\
             .And(qa.bitwiseAnd(snowBitMask).eq(0))
    
    return image.updateMask(mask)

# Function to harmonize band names
def harmonizeBands(image, bandMap):
    bandNames = list(bandMap.keys())
    newBandNames = [bandMap[key] for key in bandNames]
    return image.select(bandNames, newBandNames)

# Function to calculate indices
def calculateIndices(img):
    ndvi = img.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    evi = img.expression(
        '2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))',
        {'NIR': img.select('NIR'), 'Red': img.select('Red'), 'Blue': img.select('Blue')}
    ).rename('EVI')
    savi = img.expression(
        '((NIR - Red) / (NIR + Red + 0.5)) * 1.5',
        {'NIR': img.select('NIR'), 'Red': img.select('Red')}
    ).rename('SAVI')
    rvi = img.expression(
        'NIR / Red',
        {'NIR': img.select('NIR'), 'Red': img.select('Red')}
    ).rename('RVI')
    return img.addBands([ndvi, evi, savi, rvi]).toFloat()

# Function to filter and process Landsat images
def landsat(landsat_coll, start_date, end_date, cloud_max, bandMap):
    return landsat_coll\
        .filterDate(start_date, end_date)\
        .filterBounds(SA)\
        .filter(ee.Filter.lt('CLOUD_COVER', cloud_max))\
        .map(maskCloudAndShadows)\
        .map(lambda img: harmonizeBands(img, bandMap))\
        .map(calculateIndices)

# Function to create annual composites
def createComposite(year, month=None):
    startDate = ee.Date.fromYMD(year, month if month else 1, 1)
    endDate = startDate.advance(1, 'month' if month else 'year')
    cloud_max = 10
    
    bandMap_57 = {
        'SR_B4': 'NIR',
        'SR_B3': 'Red',
        'SR_B2': 'Green',
        'SR_B1': 'Blue'
    }
    bandMap_89 = {
        'SR_B5': 'NIR',
        'SR_B4': 'Red',
        'SR_B3': 'Green',
        'SR_B2': 'Blue'
    }
    
    if year >= 2014:
        collection = landsat(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"), startDate, endDate, cloud_max, bandMap_89)\
                    .merge(landsat(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"), startDate, endDate, cloud_max, bandMap_89))
    else:
        collection = landsat(ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"), startDate, endDate, cloud_max, bandMap_57)\
                    .merge(landsat(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"), startDate, endDate, cloud_max, bandMap_57))

    composite = collection.mean().clip(SA)\
        .set('system:time_start', startDate.millis(), 'year', year)
    
    return composite

# Function to export image to Google Drive
def exportToDrive(image, description, folder):
    task = ee.batch.Export.image.toDrive(**{
        'image': image,
        'description': description,
        'folder': folder,
        'region': SA.geometry(),
        'scale': 30,
        'maxPixels': 1e13
    })
    task.start()

def export_landsat():
    years = range(1986, 2023)

    # Iterate over each year, create composites, and export to Google Drive
    for year in years:
        annualComposite = createComposite(year)
        exportToDrive(annualComposite, 'Annual_Composite_' + str(year), 'AnnualLandSatExports_' + str(year))
