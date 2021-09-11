# Webdriver performs all the web mechanics
# - choose a browser, and link it to the browser's driver
import requests

import text2emotion as te

### Start of code

def summ_url(url):
    r = requests.get(f"https://api.smmry.com/&SM_LENGTH=5&SM_API_KEY=46265567EF&SM_URL={url}")  
    return (r.json()['sm_api_content'])

def parse_url(url):
    r = requests.get(f"https://api.smmry.com/&SM_KEYWORD_COUNT=10&SM_LENGTH=40&SM_API_KEY=46265567EF&SM_URL={url}")  
    return (r.json())

def get_image(url):
    r = requests.get(f"https://api.diffbot.com/v3/image?token=092a0a32af10cdebb4ac00f2248f6957&url={url}")
    return r.json()['objects'][4]['url']

image_url = get_image("https://www.unicef.org/emergencies/massive-earthquake-devastation-haiti")
    
summ_text = summ_url("https://www.unicef.org/emergencies/massive-earthquake-devastation-haiti")
text = parse_url("https://www.unicef.org/emergencies/massive-earthquake-devastation-haiti")

title = text['sm_api_title']
keywords = text['sm_api_keyword_array']
emotion = te.get_emotion(text['sm_api_content'])

compliedList = [title, summ_text,keywords,image_url, emotion]
print(compliedList)243238153B