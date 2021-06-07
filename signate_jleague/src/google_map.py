#!/usr/bin/env python
# coding: utf-8

# ## Google Map API Test
# 1.GeoAPIを使って住所の緯度経度を探索
# 2.PlaceAPIで商業施設、駅があるかを探索
# 3(仮)。DistrictAPIで距離と経過時間を調べる
# 
# GeoCodingAPI Doc =(https://developers.google.com/maps/documentation/geocoding/overview?hl=ja)
# 

# In[235]:


api_key = 'AIzaSyAwh_uNwyRqkadXZy-Zq8FLx56Xag9VhDQ'


# In[236]:


import requests
from urllib.parse import urlencode


# In[237]:


# from urllib.parse import urlparse, parse_qsl
# to_parse =


# In[238]:


def extract_lag_lng(address_or_postalcode, data_type = 'json'):

    end_point = f"https://maps.googleapis.com/maps/api/geocode/{data_type}"
    params = {"address": address_or_postalcode ,"key":api_key}
    url_params = urlencode(params)
    url = f"{end_point}?{url_params}"
    r = requests.get(url)
    if r.status_code not in range(200,299):
        return {}
    latlng = {}
    try:
        latlng = r.json()['results'][0]['geometry']['location']
    except:
        pass
    return latlng.get('lat'),latlng.get('lng')


# In[239]:


lat,lng = extract_lag_lng("千葉県千葉市中央区川崎町1-20")


# Places API

# In[240]:


# lat,lng = 35.5768105, 140.1229068
base_endpoint_places = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
params = {
    "key":api_key,
    "input" : "station",
    "inputtype" : "textquery"
}

locationbias = f"point:{lat},{lng}"
use_circle = True
if use_circle:
    radius = 5000
    locationbias = f"circle:{radius}@{lat},{lng}"

params['locationbias'] = locationbias

params_encoded = urlencode(params)
places_endpoint = f"{base_endpoint_places}?{params_encoded}"

print(places_endpoint)



# In[243]:


def nearbystadium(lat,lng,dist = 800 ):
    places_endpoint2 = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params_2 = {
        "key":api_key,
        "location" : f"{lat},{lng}",
        "radius" : dist, #(m) https://suumo.jp/journal/2014/05/16/62696/ より 80m 徒歩10分圏内
        "keyword" : "train station",
    }
    params_2_encoded = urlencode(params_2)
    places_url =f"{places_endpoint2}?{params_2_encoded}"

    r2 = requests.get(places_url)

    return len(r2.json()['results'])


# In[244]:

def calcdist(lat,lng,a_lat,a_lng ):
    places_endpoint2 = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params_2 = {
        "origins" : f"{lat},{lng}",
        "destinations" : f"{a_lat},{a_lng}",
        "key":api_key,
        "transit_mode":'walking'
    }
    params_2_encoded = urlencode(params_2)
    places_url =f"{places_endpoint2}?{params_2_encoded}"

    r2 = requests.get(places_url)

    return r2.json()['rows'][0]['elements'][0]['distance']['text'].split()[0]