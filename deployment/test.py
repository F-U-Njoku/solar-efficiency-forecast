import requests

solar = {'id': 0,
         'temperature': 17.618379378900883,
         'irradiance': 85.44983785023041,
         'humidity': 90.81542277591532,
         'panel_age': 13.910963039558911,
         'maintenance_count': 6.0,
         'soiling_ratio': 0.8897651368424128,
         'voltage': 6.370395942759714,
         'current': 0.0691012486940381,
         'module_temperature': 19.517274009467023,
         'cloud_coverage': 33.50988887720651,
         'wind_speed': 7.1819582155525445,
         'pressure': 1034.782455188643,
         'string_id': 'C3',
         'error_code': 'E01',
         'installation_type': 'tracking'}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=solar)
print(response.json())
