# DL-project
To recreate our results first you need to unzip the banana.zip into the same directory.
The folder Model contains 2 pretrained models (The ones we mention in section 3.6.1).
All files is setup to use the model the normal data loader.

For testing:
This will output accuracy and f1-score
simply run:
>>> python test.py

Saliency:
This will pick 10 random images and show the image along with the saliency map for that image.
simply run:
>>> python saliency.py

For training:
It is setup to train on the model named newmodel_dataloader. It will overrite one of the pretrained models.
simply run:
>>> python train.py


