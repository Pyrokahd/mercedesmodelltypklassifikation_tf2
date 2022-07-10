import pandas as pd

# Multiple Classes: Coupé, Limousine
#boolean_findings = df["Autoscout24_Karosserieform"].str.contains("Limousine")
#print(boolean_findings)


################################################
### EINLESEN DER CSV ALS BEWERTUNGSKRITERIUM ###
################################################
#typklassen
df_original = pd.read_csv("typklassen.csv", sep=";",  encoding="iso-8859-1")  # iso-8859-1 damit das mit den umlauten klappt
pd.set_option('display.max_columns', None)
#print(df.head(5))

# Spalten Bezeichnung Leerzeichen ersetzen
df_original.columns = [column.replace(" ", "_") for column in df_original.columns]
# FILL NaN bei Erstzulassung_bis, damit es keine Errors durch NaNs gibt in der Zulassungsabfrage
df_original['Erstzulassung_bis'] = df_original['Erstzulassung_bis'].fillna(2022)  # current year

#print(df.head(3))


# Wenn in der csv autoscout karosserie NaN ist => alle typen für die Query erlauben
def get_typklasse(modell, karosserieform, erstzulassung):
    """
    Nimmt modell, karosserieform und erstzulassungsjahr von autoscout 24 und gibt die Typklasse vom Fahrzeug aus.
    (speziell für Mercedes)
    :param modell: Die Modellbezeichung vom Fahrzeug
    :type modell: String
    :param karosserieform: Karosserieform laut autoscout24
    :type karosserieform: String
    :param erstzulassung: Das Jahr der Erstzulassung vom Fahrzeug
    :type erstzulassung: String
    :return: Die Typklasse vom Fahrzeug
    :rtype: String
    """
    modell_conversion_dict = {"B": "B-Klasse (alle)",
                              "C": "C-Klasse (alle)",
                              "CL": "CL (alle)",
                              "CLA": "CLA (alle)",
                              "CLK": "CLK (alle)",
                              "CLS": "CLS (alle)",
                              "E": "E-Klasse (alle)",
                              "EQC": "EQC 400",
                              "G": "G-Klasse (alle)",
                              "GL": "GL (alle)",
                              "GLA": "GLA (alle)",
                              "GLB": "GLB (alle)",
                              "GLC": "GLC (alle)",
                              "GLE": "GLE (alle)",
                              "GLK": "GLK (alle)",
                              "GLS": "GLS (alle)",
                              "M": "M-Klasse (alle)",
                              "ML": "M-Klasse (alle)",
                              "S": "S-Klasse (alle)"}
    modelle_ohne_karosserieform = ["B-Klasse (alle)", "CL (alle)", "CLS (alle)", "EQC 400", "G-Klasse (alle)",
                                   "GL (alle)", "GLA (alle)", "GLB (alle)", "GLK (alle)", "GLS (alle)",
                                   "M-Klasse (alle)"]

    df = df_original

    # modify query variables if needed
    query_modell = modell_conversion_dict[modell.split()[0]]  # z.B. C 200 => modell_dict["C"] => "C-Klasse (alle)"
    query_karosserieform = karosserieform
    query_erstzulassung = int(erstzulassung)

    # Bei CLA CLK und CLS sind Coupe und Limousine das gleiche
    if (query_modell == "CLA (alle)" or query_modell == "CLK (alle)" or query_modell == "CLS (alle)")\
            and (karosserieform == "Coupé" or karosserieform == "Limousine"):
        query_karosserieform = "Coupé, Limousine"
    # Bei GLC und GLE ist es "SUV/Geländewagen/Pickup" das ist eine Klasse bei autoscout

    #print(f"Query search parameter ({query_modell},{query_karosserieform},{query_erstzulassung})")

    #Spezialfälle für Fahrzeuge ohne feste Karosserieform in autoscout
    if query_modell in modelle_ohne_karosserieform:
        # noch mehr Spezialfälle für CLS Coupé, CLS Limousine oder CLS Kombi
        if query_modell == "CLS (alle)" and (query_karosserieform == "Coupé, Limousine" or query_karosserieform == "Kombi"):
            # normnal query für CLS coupe oder kombi
            df = df.query('Autoscout24_Modell == @query_modell and Autoscout24_Karosserieform == @query_karosserieform '
                     'and (Erstzulassung_von <= @query_erstzulassung and Erstzulassung_bis >= @query_erstzulassung)',
                     inplace=False)
        else:
            # Ansonsten die Karosserie ignorieren (weil diese häufig nicht eingetragen ist oder unterschiedlich ist)
            df = df.query('Autoscout24_Modell == @query_modell '
                     'and (Erstzulassung_von <= @query_erstzulassung and Erstzulassung_bis >= @query_erstzulassung)',
                     inplace=False)
    # Default Case für Autos ohne NaN bei Autoscout24_Karosserieform
    else:
        df = df.query('Autoscout24_Modell == @query_modell and Autoscout24_Karosserieform == @query_karosserieform '
                 'and (Erstzulassung_von <= @query_erstzulassung and Erstzulassung_bis >= @query_erstzulassung)',
                 inplace=False)

    #print(df["Typklasse"].head(3))
    #print("TypeKlasse ist: ")
    #print(df["Typklasse"].values[0])

    if len(df["Typklasse"].values) > 1:
        print(f"SHOULD NOT BE > 1 !  {df['Typklasse'].values[0]} (zweite ist {df['Typklasse'].values[1]}")

    # Return Typklasse
    if len(df["Typklasse"].values) > 0: #df.shape[0] > 0:
        return df["Typklasse"].values[0]
    else:
        print(f"ERROR: UNDEFINED CLASS ! Query search parameter ({query_modell},{query_karosserieform},{query_erstzulassung})")
        return "UNDEFINED"


#print(get_typklasse("EQC 400", "SUV/Geländewagen/Pickup", "2020"))

