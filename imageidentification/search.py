from collections import defaultdict
from duckduckgo_search import DDGS
from time import sleep
import os
import requests

class ImageSearch:
    def __init__(self, keywords:tuple=('cat', 'dog')):
        self.ddgs = DDGS()
        self.keywords = keywords
        self.urls = defaultdict(list)
    
    def search(self):
        for keyword in self.keywords:
            ddgs_images_gen = self.ddgs.images(
            keyword,
            region="wt-wt",
            safesearch="moderate",
            size=None,
            type_image=None,
            layout=None,
            license_image=None,
            max_results=10,
            )
            for image in ddgs_images_gen:
                self.urls[keyword].append(image['image'])
            sleep(10)
        return self.urls
    
    def download(self, root:str='./data'):
        os.makedirs(root, exist_ok=True)
        for keyword, urls in self.urls.items():
            for idx, url in enumerate(urls):
                response = requests.get(url)
                filename = f'{keyword}-{idx}.jpg'
                print(f'Downloading {filename}')
                with open(f'{root}/{filename}', 'wb') as f:
                    f.write(response.content)
                sleep(2)