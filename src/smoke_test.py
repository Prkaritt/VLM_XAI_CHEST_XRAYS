import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

print("numpy version: ", np.__version__)
print("pandas version: ", pd.__version__)
print("PIL image class", Image.Image)
for _ in tqdm(range(100)):
    pass