import pymongo

# open connection at port 27017 https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers
client = pymongo.MongoClient('localhost', 27017)
# create db tutorial
mydb = client["Werewolf"]
# create collection example
collection = mydb["Images"]


# Open the folder containing the card images


# random dict
post = {"authors" : ["Auteur1","Auteur2","Auteur3"],
         "title" : "Assassin's Creed",
         "affiliations" : ["University of Mannheim","University of Strasbourg","University of wonders"],
         "ref" : ["This is ref 1","This is ref 2","This is ref 3"]}
# Inserting this single dict in mongodb
collection.insert_one(post)