from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from time import sleep
from typing import List
import pandas as pd

class ImageIdentification:
    def __init__(self, path: Path) -> None:
        self.path = path
    
    def search_images(self, term: str, max_images: int):
        print(f'Searching for: {term}')
        with DDGS() as ddgs:
            results = ddgs.images(keywords=term, max_results=max_images)
            return L(results).itemgot('image')
    
    def _update_file_names(self, path:Path, prefix:str | None = None):
        for idx, file in enumerate(get_image_files(path)):
            new_file_name = '_'.join([prefix, str(idx)]) + '.jpg'
            file.rename(path/new_file_name)
                
    def download_images_from_terms(self, terms: List[str], max_images: int):
        path = self.path
        for term in terms:
            prefix = ''.join(term.split())
            download_images(path, urls=self.search_images(f'{term} photos', max_images))
            resize_image(path, max_size=400, dest=path)
            failed = verify_images(get_image_files(path))
            failed.map(Path.unlink)
            len(failed)
            self._update_file_names(path, prefix=prefix)
            sleep(10)
    
    def set_training_batch(self) -> DataBlock:
        path = self.path
        dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
            item_tfms=Resize(460),
            batch_tfms=aug_transforms(size=224)
            ).dataloaders(path)
        dls.show_batch(max_n=8)
        return dls
        
    def fine_tune_model(self, dataloaders, arch=resnet18, metrics=error_rate):
        learn = vision_learner(dataloaders, arch, metrics=metrics)
        learn.fine_tune(3)
        self.learn = learn
        
    def test_image_from_term(self, term: str):
        learn = self.learn
        file_name = term.split()[0] + '_test.jpg'
        download_url(self.search_images(term, max_images=1)[0], f'{file_name}')
        Image.open(f'{file_name}').to_thumb(256, 256)
        is_term, what, probs = learn.predict(PILImage.create(f'{file_name}'))
        print(f'This is a: {is_term}.')
        print(f'Probablity it\'s {is_term}: {probs[0]:.4f}')
        print('what is this?', what)
        
    def test_image_from_path(self, path: Path):
        learn = self.learn
        is_term, what, probs = learn.predict(PILImage.create(path))
        print(f'This is a: {is_term}.')
        print(f'Probablity it\'s {is_term}: {probs[0]:.4f}')
        print('what is this?', what)

if __name__ == '__main__':
    image_id = ImageIdentification(path=Path('maple_or_oak'))
    # image_id.download_images_from_terms(terms=['maple leaf', 'oak leaf'], max_images=100)
    dataloaders = image_id.set_training_batch()
    image_id.fine_tune_model(dataloaders)
    image_id.test_image_from_term('maple leaf')
    image_id.test_image_from_term('oak leaf')
    
        
    
