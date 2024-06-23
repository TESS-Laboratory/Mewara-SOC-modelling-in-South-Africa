# This is a simple code demo for visualising scientific interest in key terms
# (adapted by Andy Cunliffe from https://twitter.com/EShekhova/status/1772925867603165413?s=09, who adapted it from elsewhere...)
# This can be really helpful for developing quantitative summaries to evidence 
# statements like "there is rising interest in topic X"
# the europepmc package calls on the Europe PMC database, an archive of > 44 Million
# scientific outputs. While this is not complete, it’s much larger than PubMed is 
# has enough for this application. 

# Step 1 - Load packages ------------------------------------------------
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns

# Step 2 - Specify keywords ------------------------------------------------
# (e.g., categories of diet, land management practices, data sources, etc.)

keyword_soc = '''
'soil organic carbon' OR 
'soil carbon' OR 
'soil organic matter' OR 
'soil carbon stocks' OR 
'soil organic C' OR 
'soil carbon content' OR 
'organic carbon in soil' OR 
'carbon sequestration in soil' OR 
'soil carbon density' OR 
'organic soil carbon'
'''

keyword_sa = '''
'South Africa'
'''

keyword_remote_sensing = '''
'remote sensing'
'''

keyword_traditional_machine_learning = '''
'Random forest' OR 'Support Vector Machine'
'''

keyword_deep_learning = '''
'Deep Learning' OR 
'Convolutional Neural Network' OR 
'Deep Neural Network' OR 
'Neural Network'
'''

keyword_soc_sa = f'({keyword_soc}) AND ({keyword_sa})'

keyword_soc_sa_remote_sensing = f'{keyword_soc_sa} AND ({keyword_remote_sensing})'

keyword_soc_sa_ML = f'{keyword_soc_sa} AND ({keyword_traditional_machine_learning})'

keyword_soc_sa_DL = f'{keyword_soc_sa} AND ({keyword_deep_learning})'

keyword_soc_sa_remote_sensing_DL = f'{keyword_soc_sa_remote_sensing} AND ({keyword_deep_learning})'

keyword = keyword_soc_sa_DL

# Step 3 - Fetch metrics ------------------------------------------------
# Use the europe_pmc package to query the Europe PubMed Central RESTful Web Service to 
# fetch the number of publications with specific keywords for each year. 
def fetch_data(query):
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        'query': query,
        'resultType': 'lite',
        'synonym': 'true',
        'format': 'json',
        'pageSize': 1000,
        'cursorMark': '*'
        }

    all_results = []
    while True:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            result = response.json()
            papers = result['resultList']['result']
            all_results.extend(papers)
            if 'nextCursorMark' in result and papers != []:
                params['cursorMark'] = result['nextCursorMark']
            else:
                break
        else:
            print(f"Error fetching data: {response.status_code}")
            break

    return all_results

def fetch_trends(query, start_year, end_year):
    papers = fetch_data(query)
    year_counts = defaultdict(int)
    for paper in papers:
        if 'pubYear' in paper:
            year = paper['pubYear']
            if start_year <= int(year) <= end_year:
                year_counts[year] += 1

    return pd.DataFrame({'year': list(year_counts.keys()), 'query_hits': list(year_counts.values())})

# Fetch metrics
trend = fetch_trends(query=keyword, start_year=2000, end_year=2023)
trend['keywords'] = keyword

# Step 4 - Combine data ------------------------------------------------

combined_data = pd.concat([trend]).sort_values(by = 'year')

# Step 5 - Visualise data ------------------------------------------------

plt.figure(figsize=(14, 10))
sns.barplot(data=combined_data, x='year', y='query_hits', hue='keywords', palette='viridis')
plt.title('Scientific Interest in Studying SOC')
plt.xlabel('Year')
plt.ylabel('Number of Published Papers')
plt.xticks(rotation=60)
plt.legend(title='Query')
plt.tight_layout()
plt.show()
print('Enter to exit')
