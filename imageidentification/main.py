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
    
    def download_images_from_terms(self, terms: List[str], max_images: int):
        path = self.path
        for term in terms:
            folder = ''.join(term.split())
            dest = (path/folder)
            dest.mkdir(exist_ok=True, parents=True)
            download_images(dest, urls=self.search_images(f'{term} photos', max_images))
            resize_image(path/folder, max_size=400, dest=path/folder)
            failed = verify_images(get_image_files(path))
            failed.map(Path.unlink)
            len(failed)
            sleep(10)
    
    def set_training_df(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=['fname', 'labels', 'is_valid'])
        path = self.path
        for dir in os.listdir(path):
            if os.path.isdir(path/dir):
                for file in os.listdir(path/dir):
                    if os.path.isfile(path/dir/file):
                        df = df._append({'fname': file, 'labels': dir, 'is_valid': True}, ignore_index=True)
        self.df = df
        return df
    
    def set_training_batch(self) -> DataBlock:
        path = self.path
        df = self.df
        training_batch = DataBlock(blocks=(ImageBlock, CategoryBlock),
                        splitter=ColSplitter('is_valid'),
                        get_x=ColReader('fname', pref=str(path) + os.path.sep),
                        get_y=ColReader('labels', label_delim=' '),
                        item_tfms = Resize(460),
                        batch_tfms=aug_transforms(size=224)).dataloaders(df)
        training_batch.show_batch(max_n=8)
        return training_batch
        
    def fine_tune_model(self, training_batch: DataBlock, arch=resnet18, metrics=error_rate):
        learn = vision_learner(training_batch, arch, metrics=metrics)
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
    image_id.set_training_df()
    training_batch = image_id.set_training_batch()
    image_id.fine_tune_model(training_batch)
    image_id.test_image_from_term('maple leaf')
    image_id.test_image_from_term('oak leaf')
    
        
    
