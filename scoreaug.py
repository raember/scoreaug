from pathlib import Path
from random import choice
from typing import List, Generator, Union

import numpy as np
from PIL.Image import Image, open as img_open


def augment_score(img: Union[Image, np.ndarray], bgs: List[Image]) -> Image:
    if isinstance(img, Image):
        img = np.array(img)
    bg_img = choice(bgs)
    bg = np.array(bg_img.resize(img.shape[:2][::-1]))
    return np.minimum(img, bg)


def get_bg_imgs(bg_im_path: str = '../imslp/imslp') -> Generator[Image]:
    for f in Path(bg_im_path).glob('*'):
        yield img_open(f)
