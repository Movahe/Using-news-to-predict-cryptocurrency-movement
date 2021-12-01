from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

def read_api_key(filepath):
    """
    Read the Yelp API Key from file.

    Args:
        filepath (string): File containing API Key
    Returns:
        api_key (string): The API Key
    """

    # feel free to modify this function if you are storing the API Key differently
    with open(filepath, 'r') as f:
        api_key = f.read().replace('\n', '')
#        print(api_key)
        return api_key



API_HOST = 'https://pro-api.coinmarketcap.com'
SEARCH = '/v1/cryptocurrency/bitcoin/historical'

url = API_HOST + SEARCH

parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD',
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': read_api_key('cryptocurrency.txt'),
}

session = Session()
session.headers.update(headers)

try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
    print(data)
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)

