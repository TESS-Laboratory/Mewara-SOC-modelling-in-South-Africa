import os
import requests
import json

# get Landsat scenes using token authentication
def search_landsat_scenes(api_token, bbox, start_date, end_date, max_cloud_cover):
    url = 'https://m2m.cr.usgs.gov/api/api/json/stable/search'
    headers = {'X-Auth-Token': api_token}
    payload = {
        "datasetName": "LANDSAT_8_C1",
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {"latitude": bbox[1], "longitude": bbox[0]},
            "upperRight": {"latitude": bbox[3], "longitude": bbox[2]}
        },
        "temporalFilter": {
            "startDate": start_date,
            "endDate": end_date
        },
        "maxCloudCover": max_cloud_cover,
        "includeUnknownCloudCover": False
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['data']['results']

# download a scene
def download_scene(scene_id, api_token, output_dir):
    url = f'https://m2m.cr.usgs.gov/api/api/json/stable/download'
    headers = {'X-Auth-Token': api_token}
    payload = {
        "entityIds": [scene_id],
        "datasetName": "LANDSAT_8_C1",
        "products": ["STANDARD"]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    download_url = response.json()['data'][0]['url']
    local_filename = os.path.join(output_dir, scene_id + '.zip')
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

api_token = 'wLSQAwWPEr7Ex!9UpJIgbc_SnPdBNNBVrpjWvpY_y7SMS8d0FyedCldP48GHi@Kd'

# South Africa bounding box
south_africa_bbox = [16.344976, -34.819166, 32.83012, -22.125423]

start_date = '1986-01-01'
end_date = '2023-12-31'

# max cloud cover 10%
max_cloud_cover = 10

def download_landsat_images():
    scenes = search_landsat_scenes(api_token, south_africa_bbox, start_date, end_date, max_cloud_cover)

    print(f"Found {len(scenes)} scenes.")

    # Directory to save images
    output_dir = f'Data\LandSat'
    os.makedirs(output_dir, exist_ok=True)

    # Download scenes
    for scene in scenes:
        scene_id = scene['entityId']
        print(f"Downloading scene {scene_id}...")
        download_scene(scene_id, api_token, output_dir)

    print("Download completed.")

download_landsat_images()

