import gdown
import zipfile
url = 'https://drive.google.com/uc?id=1pA7ZeXU3hh-1txS9lFQiCek1ts3MdBaj'
out = 'data/jhu-v2.zip'
gdown.download(url, out, quiet=False)
with zipfile.ZipFile(out, 'r') as zip_ref:
    zip_ref.extractall('data/eset')