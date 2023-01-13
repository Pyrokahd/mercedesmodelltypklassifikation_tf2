# mercedesmodelltypklassifikation_tf2
A CNN made with TF2 to classify images of mercedes-benz cars and detect what type of car it is.

In this project the dataset was created by a webcrawler and then cleaned by object detection and manual sorting.
The model itself is a finetuned resnet50v2.

(Structure description in german)

Falls Modell_v7 geladen werden soll, muss es zuvor wie in der Create_models.py Datei gezeigt, erstellt werden. Parameter sind: resnetType = 'ResNet50V2', dropoutrate = 0.2 und base_layer_to_finetune = 40.

# Struktur
Die Gruppierung und Struktur der einzelnen Dateien und Ordner ist hier dargestellt. <br/>
Es gibt zwei Funktionalitäten, den Webcrawler und den Deep Learning Klassifizierer. Utility Skripte werden genutzt um die Daten z.B. vorverarbeiten durch Augmentierung oder Aufteilen in Train- und Test/-set. Resultate der Experimente sind im resultdata Ordner zu finden. <br/>
<br/>
![filtered images by algorithm](/doc_assets/ProjektStrukturDiagramm.drawio.png)

# Description
## Step 1: Getting data
In order to create a dataset, a web crawler was built in python to get images from autoscout24 as well as the truth label based on the description. <br/>
To filter out many useless images a pretrained object detection model was used to detect if a car is in the image. An example of such filtered images is below:
![filtered images by hand](/doc_assets/filteredImagesByAlgo.PNG)
<br/>
Some images could not be filtered through that method since there are cars visible but either not the wanted care or its another car or a 3D Render / Image of a car at a screen i.e. See below:
![Projekt Struktur Gruppierung](/doc_assets/filteredImagesByHand.PNG)
Those are then filtered by hand. However through clustering the images into groups and if those images fit one cluster they could be sorted out automated in theory. Though this experiment didn't fit within the timeframe of this project.
<br/>

## Step 2: Training a model
For training a pretrained ResNet is used with an added classifier and further finetuned. 149.874 images are used to classify 65 classes. Imbalanced classes are dealt with through augmentation and class weights.
The archieved performance on the test set is 86,66% and performance per class is shown below:
![accuracy metrics](/doc_assets/Accuracy.PNG)
Images are of size 225x224 pixels and there are (after augmentation) 143976 images in the trainset, 15997 images in the validation set and 27801 images in the testset. All sets sharing the same distribution per class.
<br/>

## Step 3: Evaluate wrong predictions
To check what is holding back the performance, the most common cases of wrong classifications are looked at. Then it is checked wheter the significant feature to differentiate the car type is even visible in the images. The result was that in 624 images that where looked at, 521 didn't show the relevant car features and thus not even a human could correctly identify this car type. <br/>
Another intersting case was one specific class that got correctly classified by the model but had a wrong truthlabel to it and thus contributed to wrong predictions. The image below is the car type in question (C292). Almost all of those had the wrong truth label (W166) because people in autoscout24 wrongly describe it as SUV even though it is a Coupé.
![Wrongly Classified images](/doc_assets/W166correctlyPredictedAsC292.PNG) 
