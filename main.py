from pathlib import Path

from PIL.Image import open as img_open

from scoreaug import get_bg_imgs, augment_score

style_img_path = Path('../imslp/imslp')
ds2_img_path = Path('../ds2_dense/images')

assert style_img_path.exists()
assert ds2_img_path.exists()

bg_imgs = list(get_bg_imgs(str(style_img_path)))
for f in ds2_img_path.glob('*'):
    augmented = augment_score(img_open(f), bg_imgs)
    break
