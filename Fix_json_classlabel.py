"""
This Script is only used to fix the missing class labels in the metadata.json.
this is done by iterating over the images and comparing the id of the Image to the id in the json.
If its the same, the class from the image is added
"""

import json
import os
import time

filter_from_path = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"
filePath = "C:/Users/Christian/PycharmProjects/InnovationsProjektWebScraper/metaDaten.json"

# TODO vor dem einlesen müssen klammern vor und nach dem json-file/string hinzugefügt werden [] -- wurde in der file direkt gemacht (von hand)
jsonfile = open(filePath, "r", encoding="utf-8")
#jsonstring = jsonfile.read()
json_objects = json.load(jsonfile)
jsonfile.close()

print(len(json_objects))
# have a look
#print(json_objects[0])
#firstJson = json_objects[0]  # converts it into python dict
#print(firstJson)
#print(firstJson["id"])
#print(type(firstJson["Erstzulassung"]))
#
#test = {"json":"è,é"}
#json_string = json.dumps(test, ensure_ascii=False)
#print(json_string)
#
# if there are still decode errors https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c

for root, dirs, files in os.walk(filter_from_path, topdown=True):
    t1 = time.perf_counter()

    if len(files) > 0:  # im ersten Verzeichnis sind keine files sonder nur weitere Ordner
        for file in files:
            id = int(str(file).split("-")[1])
            # example path: C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt\S204
            # os.walk uses os.sep for subfolders thats why the last one has "\" and not "/"
            category = root.split(os.sep)[-1]
            for object in json_objects:
                #print(object["Karosserieform"])
                if object["id"] == id:
                    object["label"] = category
        # search object with id
        # add the class category

    t2 = time.perf_counter()
    if len(files) > 0:
        print(f"1 Folder checked: {t2 - t1} sec: {file}")

print(len(json_objects))
#jsonfile = open(filePath, "w")
#json.dump(json_objects, jsonfile, indent=2)
#jsonfile.close()