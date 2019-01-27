import os
import sys
from PIL import Image
from pred import *

input_dir = "./faces/"
model_path = "face_model.pkl"

with open(model_path) as f:
    clf, labels = pickle.load(f)

threshold = 0.8

file_ethnicities = {}

i = 0

for fname in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, fname)
    try:
        pred, locs = predict_one_image(img_path, clf, labels)
    except:
        print("Skipping {}".format(img_path))
    classification = pred.values[0]
    gender = classification[0]
    asian = classification[1]
    white = classification[2]
    black = classification[3]
    if max(white, asian, black) > threshold:
        if asian > white and asian > black:
            file_ethnicities[fname] = "asian"
        elif white > asian and white > black:
            file_ethnicities[fname] = "white"
        elif black > white and black > asian:
            file_ethnicities[fname] = "black"

    if i > 5:
        break
    else:
        i = i + 1

print(file_ethnicities)