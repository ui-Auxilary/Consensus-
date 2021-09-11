from flask import Flask, render_template, request, redirect
from collections import OrderedDict
import numpy as np

import requests

import text2emotion as te

app = Flask(__name__)


def parse_url(url):
    r = requests.get(f"https://api.smmry.com/&SM_KEYWORD_COUNT=10&SM_LENGTH=5&SM_API_KEY=243238153B&SM_URL={url}")  
    return (r.json())

def get_image(url):
    r = requests.get(f"https://api.diffbot.com/v3/image?token=092a0a32af10cdebb4ac00f2248f6957&url={url}")
    max_num = len(r.json()['objects'])
    if (max_num > 2):
      return r.json()['objects'][max_num - 2]['url']
    else:
      return r.json()['objects'][0]['url']


title = "Joint COVAX statement on supply forecast for 2021 and early 2022"
summary = "With the support of the international community, COVAX immediately began securing financing, entering into negotiations with vaccine developers and manufacturers and addressing the host of technical and operational challenges associated with rolling out the largest and most complex vaccination programme in history."
image = "https://www.unicef.org/sites/default/files/styles/press_release_feature/public/vacunacion-covax-peru-historia2.jpg?itok=U41Dw0fS"
emotion = {'Angry': 0.12, 'Fear': 0.42, 'Happy': 0.04, 'Sad': 0.33, 'Surprise': 0.08}
keywords = ['Keyword 1', 'Keyword 2', 'Keyword 3', 'Keyword 4', 'Keyword 5']

@app.route('/')
def home():
    return render_template('searchpage.html', title=title, summary=summary, image=image, emotion=emotion, keywords=keywords)

@app.route('/form', methods=['POST'])
def form():
    urlpath = request.form['projectform']
    print(urlpath)

    image_url = get_image(urlpath)
    text = parse_url(urlpath)
   
    summ_text = text['sm_api_content']

    title = text['sm_api_title']
    keywords = text['sm_api_keyword_array']
    emotion = te.get_emotion(text['sm_api_content'])

    return render_template('poster.html', title=title, summary=summ_text, image=image_url, emotion=emotion, keywords=keywords)



app.run(host='0.0.0.0')