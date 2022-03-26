import zipfile

with zipfile.ZipFile("data/train.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")

with zipfile.ZipFile("data/test.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")