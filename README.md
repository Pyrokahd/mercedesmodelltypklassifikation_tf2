# mercedesmodelltypklassifikation_tf2
A CNN made with TF2 to classify images of mercedes-benz cars and detect what type of car it is.

In this project the dataset was created by a webcrawler and then cleaned by object detection and manual sorting.
The model itself is a finetuned resnet50v2.

(Description in german)

Falls Modell_v7 geladen werden soll, muss es zuvor wie in der Create_models.py Datei gezeigt, erstellt werden. Parameter sind: resnetType = 'ResNet50V2', dropoutrate = 0.2 und base_layer_to_finetune = 40.

# Struktur
Die Gruppierung und Struktur der einzelnen Dateien und Ordner ist hier dargestellt. <br/>
Es gibt zwei Funktionalitäten, den Webcrawler und den Deep Learning Klassifierer. Utility Skripte werden genutzt um die Daten z.B. vorverarbeiten durch Augmentierung oder Aufteilen in Train- und Test/-set. Resultate der Experimente sind im resultdata Ordner zu finden. <br/>
<br/>
![Projekt Struktur Gruppierung](/doc_assets/ProjektStrukturDiagramm.drawio.png)
