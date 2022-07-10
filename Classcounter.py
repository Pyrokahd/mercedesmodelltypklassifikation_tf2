"""
Counts the Images per Class in the given Data Folder and plots it as horizontal bar chart.
"""

import os
import matplotlib.pyplot as plt

filter_from_path = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"

class_count = {}

for root, dirs, files in os.walk(filter_from_path, topdown=True):
    counter = 0
    category = root.split(os.sep)[-1]
    if len(files) > 0:
        for file in files:
            counter += 1

    # To not include the root folders as classes with 0 files
    if category != "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"\
            and category != "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt_TEST_SET":
        class_count[category] = counter

for key, value in class_count.items():
    print(f"Category: {key}, Anzahl {value}")


################
# PLOT A GRAPH #
################

plt.rcdefaults()

fig, ax = plt.subplots(figsize=(16, 6))

# Example data
labels = []
for key in class_count.keys():
    labels.append(key)

sum = 0
values = []
for value in class_count.values():
    values.append(value)
    sum += value

print(f"Daten insgesamt: {sum} Bilder")

plt.axhline(y=500, color='grey', linestyle='-.', lw=0.6)
plt.axhline(y=1000, color='grey', linestyle='-.', lw=0.6)
plt.axhline(y=1500, color='grey', linestyle='-.', lw=0.6)
plt.axhline(y=2000, color='grey', linestyle='-.', lw=0.6)
plt.axhline(y=2500, color='grey', linestyle='-.', lw=0.6)

ax.bar(labels, values, align='center')
#ax.invert_yaxis()  # labels read top-to-bottom
plt.xticks(rotation=50)
ax.set_xlabel('Samples per Class')
ax.set_title('Pictures per class in Train_set')
plt.savefig("resultdata/Data_Count_Train.png")
plt.show()