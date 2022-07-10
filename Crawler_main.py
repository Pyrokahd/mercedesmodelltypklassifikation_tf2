"""
The Main Script that does the Web Crawling by iterating through autoscout24.
The class is defined by the crawled meta data and a comparison using the typclasses in the csv file.
"""


import requests
import urllib.request
from bs4 import BeautifulSoup
import json
import Crawler_helper
import logging
import os
import pandas as pd


##############
# Set Logger #
logfile = 'debug.log'
open(logfile, 'w')
logging.basicConfig(filename=logfile, level=logging.INFO)
##############

SAVE_TO_URL = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt_crawling"

def get_next_sibling(element):
    """
    Similar to Beautifuls next_sibling
    Custome get sibling to ignore empty lines and return the next sibling instead
    :param element: BeautifulSoup Object to be searched
    :type element: BeautifulSoup Object
    :return: returns the next sibling
    :rtype: BeautifulSoup Object
    """
    sibling = element.next_sibling
    if sibling == "\n":
        return get_next_sibling(sibling)
    else:
        return sibling


# open all already searched URLs to not crawl them again.
try:
    with open("searched_urls.txt") as file:
        searched_urls = []
        for line in file:
            line = line.replace("\n", "")  # Strip line breaks
            searched_urls.append(line)
# falls die file nicht existiert  (wird dann später im code generiert wenn die ersten links gespeichert werden)
except:
    print("couldnt open file")
    searched_urls = []
print(f"searched URLs to begin with: {len(searched_urls)}")


def crawl_400_proposals(marke, klasse, bisKM, vonKM, bisZulassung, vonZulassung, karosserieform):
    """
    Takes search parameter, searches autoscout24 for cars and saves all pictures with correct class to defined path
    and adds other found information to the "metaDaten.json" (placed in this working directory).
    :param marke:
    :type marke:
    :param klasse:
    :type klasse:
    :param bisKM:
    :type bisKM:
    :param vonKM:
    :type vonKM:
    :param bisZulassung:
    :type bisZulassung:
    :param vonZulassung:
    :type vonZulassung:
    :param karosserieform:
    :type karosserieform:
    :return:
    :rtype:
    """
    global SAVE_TO_URL
    # Loop über alle 20 Angebote pro Seite (Es sind immer maximal 20 pro Seite bei Autoscout)
    for p in range(1, 21):
        # Hier im Link Marke und Klasse
        # Wichtig! Kein ocs listing (von autoscout selbst)! Diese haben ein anderes Format und lassen sich nicht crawlen
        # wir wollen auch kein leasing (hat ebenfalls anderes format) also: hasleasing=false und ocs_listing=exclude
        URL_with_options = "https://www.autoscout24.de/lst/"+marke+"/"+klasse +\
                           "?ocs_listing=exclude&sort=standard&desc=1&hasleasing=false&ustate=N%2CU&size=20&page="+str(p)\
                            + f"&kmto={bisKM}&kmfrom={vonKM}" + f"&fregto={bisZulassung}&fregfrom={vonZulassung}" \
                           + f"&body={karosserieform}"
        #URL_with_options = ""

        #print(URL_with_options)
        page = requests.get(URL_with_options)
        soup = BeautifulSoup(page.content, "html.parser")

        ### ALLE GELISTETEN ANGEBOTE PRO SEITE ###
        angebote = soup.find_all("div", class_="cl-list-element cl-list-element-gap")
        alle_20_angebote = []
        # hier parent von parent sollte nicht die klasse haben
        for angebot in angebote:
            # Nur wenn auch Angebote gefunden wurden, alternativ Vorschläge sollen ignoriert werden (Recommendations)
            if not angebot.parent.parent.get("class") == "cl-list-element-recommendations":
                # DORT IM TITEL DEN LINK ZUM ANGEBOT SUCHEN
                titelDiv = angebot.find(class_="cldt-summary-titles")
                if titelDiv is not None:
                    angebots_link = titelDiv.find("a")["href"]

                    # LINK VERVOLLSTÄNDIGEN # Wenn er nicht bereits mit autoscout beginnt (ist bei einigen der Fall)
                    if not angebots_link[0:26] == "https://www.autoscout24.de":
                        angebots_link = "https://www.autoscout24.de" + angebots_link
                    alle_20_angebote.append(angebots_link)

        for link in alle_20_angebote:
            if link in searched_urls:
                print(link)
                print("ES IST BEREITS GESEARCHED")

        alle_20_angebote = [link for link in alle_20_angebote if link not in searched_urls]
        for link in alle_20_angebote:
            searched_urls.append(link)
            # searched URLs speichern
            with open("searched_urls.txt", "a") as file:
                file.write(link+"\n")

            #### FÜR JEDEN LINK/JEDES ANGEBOT ###
            # Versuche Angebotsseite zu holen:
            # wenn es nicht klappt weiter zum nächsten angebotslink
            try:
                print("link to request")
                print(link)
                page = requests.get(link)
                soup = BeautifulSoup(page.content, "html.parser")


                ##################
                #### MERKMALE ####
                ##################
                # Merkmal tabelle suchen (nächstes Element nach der H3 Überschrift Merkmale)
                merkmale = soup.find("h3", class_="sc-font-bold", string="Merkmale")

                merkmal_tabelle = get_next_sibling(merkmale)

                # Relevante Merkmale Bestimmen
                #auszulesende_merkmale = ["Marke", "Modell", "Erstzulassung", "Karosserieform", "Außenfarbe"
                #                         , "Innenausstattung", "Anzahl Türen", "Sitzplätze", "Schlüsselnummer"]

                merkmal_überschriften = merkmal_tabelle.find_all("dt")
                # Merkmalüberschrift und Merkmalwert auslesen:
                merkmal_dict = {}
                for h in merkmal_überschriften:
                    #if h.text.strip() in auszulesende_merkmale:
                    value_element = get_next_sibling(h)
                    merkmal_dict[h.text.strip()] = value_element.text.strip()
                ################
                #### BILDER ####
                ################

                all_image_elements = soup.find_all("img", class_="gallery-picture__image")
                # SKIP Beiträge ohne Bilder
                if all_image_elements is not None and len(all_image_elements) > 1:
                    all_image_links = []

                    nummer = 0
                    for image in all_image_elements:
                        # Das active element nutzt src die anderen data-src, also src nehmen wenn möglich sonst data-src
                        try:
                            all_image_links.append(image['src'])
                        except:
                            all_image_links.append(image['data-src'])
                        finally:
                            #print(f"")
                            continue

                    for ilink in all_image_links:
                        #print(ilink)
                        #print(f"Input: {merkmal_dict['Modell']}, {merkmal_dict['Karosserieform']}, {merkmal_dict['Erstzulassung']}")

                        id = len(searched_urls)
                        typKlasse = Crawler_helper.get_typklasse(merkmal_dict["Modell"], merkmal_dict["Karosserieform"],
                                                                 merkmal_dict["Erstzulassung"])

                        # sonderfall W461/463 (Slash entfernen) weil sonst kein Ordner erstellt werden kann
                        typKlasse = typKlasse.replace("/", "-")

                        name = "id-"+str(id)+"-image"+str(nummer)+"-"+typKlasse+".jpg"

                        # Check if a folder for this class exists, create one if not
                        if not os.path.exists(SAVE_TO_URL+"/"+typKlasse):
                            os.mkdir(SAVE_TO_URL+"/"+typKlasse)
                        # save the image to the class folder it belongs to
                        try:
                            urllib.request.urlretrieve(ilink, SAVE_TO_URL+"/"+typKlasse+"/"+name)
                            print("Image saved")
                        except ValueError:
                            logging.error(ValueError)
                            print(ValueError)
                            print(f"Couldn´t save from URL, probably no Image (video or 360° Image)")

                        nummer += 1
                else:
                    print("DAS WAR NIX! (Keine Bilder für das Angebot oder nur 1")

                ######################
                ### SAVE META DATA ###
                ######################
                # Id zu den Merkmalen hinzufügen
                merkmal_dict["id"] = len(searched_urls)
                merkmal_dict["url"] = link
                merkmal_dict["label"] = typKlasse

                # Json eintrag erstellen
                with open("metaDaten.json", "a", encoding='UTF-8') as file:
                    json.dump(merkmal_dict, file, indent=2, ensure_ascii=False)
                    file.write(",")
                    # TODO replace specials signs like ß and é (in keys and values of the merkmal_dict)
                    # NOTE brackets "[" and "]" are missing in this file since every json object is saved on its own
                    # also at the end there is one "," to much.

            except ConnectionError:
                logging.error(ConnectionError)
                print(ConnectionError)
                print(f" couldnt request the link: {link}")


# Karosserieformen als REST Variablen
# multiple are combined with [Number]%2C[Number]
karosserieform_to_queryparameter = {
    "Kleinwagen": "1",
    "Cabrio": "2",
    "Coupé": "3",
    "SUV/Geländewagen/Pickup": "4",
    "Limousine": "6",
    "Kombi": "5",
    "Transporter": "13",
    "Van/Kleinbus": "12",
    "Sonstige": "7",
    "Coupé, Limousine": "3%2C6",
    "nan": ""
}

#################################
### Search parameter from csv ###
#################################
#typklassen
df = pd.read_csv("typklassen.csv", sep=";",  encoding="iso-8859-1")  # iso-8859-1 damit das mit den umlauten klappt
# Spalten Bezeichnung Leerzeichen ersetzen
df.columns = [column.replace(" ", "_") for column in df.columns]
# FILL NaN bei Erstzulassung_bis, damit es keine Errors durch NaNs gibt in der Zulassungsabfrage
df['Erstzulassung_bis'] = df['Erstzulassung_bis'].fillna(2021)  # current year (Zum Zeitpunkt dieser Arbeit)

marke = "mercedes-benz"

# Um spezielle Typklassen zu suchen (Dafür ein simples if Statement in df.iterrows) Zeile 265
classes_to_search_again = ["X167", "S202", "C216"]

if True:
    step = 5000
    # Aktuell rückwärts um die alten autos zu finden
    for km in range(0, 250000, step):   # range(200000, 0, -5000) # for backward iteration
        #Kilometerstand in 5000er Schritten (Um die Anfrage einzuschränken weil maximal 400 gezeigt werden)
        vonKM = str(km)
        bisKM = str(km+step)
        # Suchparameter auslesen und damit eine Abfrage an Autoscout senden und die Daten sammeln
        for index, row in df.iterrows():

            # Um nach speziellen Typklassen zu suchen kann der folgende Code in dieses If-Statement gesetzt werden
            # und der folgende code bis zum print(40 * "-") eingerückt werden
            # if row["Typklasse"] in classes_to_search_again:

            klasse = str(row["Autoscout24_Modell"])
            # klasse bearbeiten
            klasse = klasse.lower()
            klasse = klasse.replace(" ", "-")

            bisZulassung = str(int(row["Erstzulassung_bis"]))
            vonZulassung = str(int(row["Erstzulassung_von"]))
            form = str(row["Autoscout24_Karosserieform"])
            # Umwandeln der Karosserieform von der CSV zu dem Parameter für den link  (nan is handled in the dict)
            karosserieform = karosserieform_to_queryparameter[form]


            print(f"Parameter: {marke}, {klasse}, KM von: {vonKM}, KM bis: {bisKM}, Zulassung von: {vonZulassung},"
                  f" Zulassung bis: {bisZulassung}, {karosserieform}")

            # Crawl the Data:
            crawl_400_proposals(marke, klasse, bisKM, vonKM, bisZulassung, vonZulassung, karosserieform)

            print(40 * "-")
            print("400 done")
            print(40 * "-")



