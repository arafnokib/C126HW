import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import ssl, os
import pandas as pd

#Setting an HTTPS Context to fetch data from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']

y = pd.read_csv("labels.csv")["labels"]

##X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
classes = ['A', 'B', 'C','D', 'E','F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)
def get_prediction(image):
    
        im_pil = Image.open(image)
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
        
        pixel_filter = 20 
        min_pixel = np.percentile(image_bw_resized, pixel_filter) 
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255) 
        max_pixel = np.max(image_bw_resized) 
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784) 
        test_pred = clf.predict(test_sample) 
        return test_pred[0]   