#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the Base object used in LISC
from lisc.objects.base import Base

# Initialize a base object
base = Base()

# Import a helper function to create a LISC file structure
from lisc.utils.db import create_file_structure

# Create a database file structure
db = create_file_structure()


# In[2]:


# Import the Words object, which is used for words collection
from lisc import Words

# Import the SCDB object, which organizes a database structure for saved data
from lisc.utils.db import SCDB

# Import a utility function for saving out collected data
from lisc.utils.io import save_object




# Import the custom objects that are used to store collected words data
from lisc.data import Articles, ArticlesAll

# Import database and IO utilities to reload our previously collected data
from lisc.utils.db import SCDB
from lisc.utils.io import load_object

# Import plots that are available for words data
from lisc.plts.words import plot_wordcloud

import scispacy
import spacy
nlp = spacy.load("en_ner_bionlp13cg_md")



import matplotlib.pyplot as plt
from wordcloud import WordCloud


import pandas as pd
import sys


# In[3]:


# Set some search terms
terms = [['roundworm','Ascaris lumbricoides', 'Ascariasis'], 
         ['whipworm','Trichuris', 'trichiura'], 
         ['hookworm','Nector', 'americanus', 'Ancylostoma', 'duodenale']]


# In[4]:


# Initialize Words object and set the terms to search for
words = Words()
words.add_terms(terms)

infant_grp = ['infant', 'infants']
child_grp = ['child', 'children']
adult_grp = ['adult', 'adults']

age_grp = "child"

# age_grp = sys.argv[2]

if age_grp == "adult":
    age_group_in = adult_grp
    age_group_ex = infant_grp+child_grp
    
elif age_grp == "infant":
    age_group_in = infant_grp
    age_group_ex = adult_grp+child_grp
    
elif age_grp == "child":
    age_group_in = child_grp
    age_group_ex = infant_grp+adult_grp

term_list = ["roundworm", "whipworm", "hookworm"]

symptoms_list = ['anaemia', 'diarrhoea', "cough", "eosinophilic",
                 "loeffler", "hepatopancreatic", "dyspnoea", "haemoptysis", "wakana", 
                 "asthenia", "abdominal", "pain", "oedema", "occult", "faecal", "blood", 
                 "melaena", "appetite", "gastrointestinal", "bleeding", 
                 "bowel", "obstruction", "volvulus", "intussusception", "peritonitis", 
                 "gastric", "rectal", "prolapse", "microbiome", "microbiota", "gut", 
                 "metabolites", "anemia", "intestinal", "growth", "faltering", 
                 "vomit", "volmiting", "cholangitis", "pancreatitis", "anorexia", "gall", "gallbladder", 
                 "bladder", "cough", "cancer", "developmental", "hyperactivity"]

geography_list = ["africa", "america", "asia"]

microbiome_list = ["bacteriodes","fermicutes", "proteobacteria", "actinobacteria", 
                  "oscillibacter", "flavonifractor", "butyrivibrio", "allobaculum", 
                   "solobacterium", "lactobacillus", "campylobacter", "heligmosomoides"]

immune_list = ["immunity", "inflammation", "allergy", 
               "chronicity",   "Tregs", "Th2", "IL33", "IL25",  "cytokines", "protection", 
               "vaccines", "benefits", "IgE", "eosinophils", "sensitization"]

inclustion_terms = ['symptoms', 'symptom'] + age_group_in
exclustion_terms = age_group_ex

# Set up inclusions and exclusions
#   Each is a list, that should be the same length as the number of terms
inclusions = [inclustion_terms, 
              inclustion_terms, 
              inclustion_terms]
exclusions = [exclustion_terms, 
              exclustion_terms, 
              exclustion_terms]


# In[5]:


num_article = 5000
num_list = 1

# num_article = int(sys.argv[1])
# num_list = int(sys.argv[3])

if num_list == 1:
    chosen_list = symptoms_list
elif num_list == 2:
    chosen_list = geography_list
elif num_list == 3:
    chosen_list = microbiome_list
elif num_list == 4:
    chosen_list = immune_list
    
    


# In[6]:


# Collect words data
words.run_collection(retmax=num_article)


# In[7]:


# Set up our database object, so we can save out data as we go
db = SCDB('lisc_db')

# Collect words data
words.run_collection(usehistory=True, retmax=num_article, save_and_clear=True, directory=db)


# In[8]:


# Save out the words data
save_object(words, 'tutorial_words', directory=db)


# In[9]:


# Reload the words object, specifying to also reload the article data
words = load_object('tutorial_words', directory=SCDB('lisc_db'), reload_results=True)

# Preprocess article data
words.process_articles()

# Process collected data into aggregated data objects
words.process_combined_results()


# In[10]:


words_list = []

print("Age Group:", age_grp)

for num in range(0,3):
    words_dict = {}
    print("----------------------------------", term_list[num],"-------------------------------")
    for i in words.combined_results[num].words:
        doc = nlp(i)
        for token in doc:
            if(token.pos_ == "NOUN" or token.pos_ == "ADJ"):
                if(words.combined_results[num].words[i] > 1 and i in chosen_list):
                    print(i, "-", words.combined_results[num].words[i], "[", token.ent_type_,"]")
                    words_dict[i] = words.combined_results[num].words[i]
    
    words_list.append(words_dict)
                    
    print()
    print()
                    


# In[11]:


import pandas as pd
# import dataframe_image as dfi

for i in range(0,3):
    data_items = words_list[i].items()
    data_list = list(data_items)
    df = pd.DataFrame(data_list)
    
#     dfi.export(df, 'img/{}_dataframe_{}_{}.png'.format(age_grp,term_list[i], num_list))
    df.to_csv('img/{}_dataframe_{}_{}.csv'.format(age_grp,term_list[i], num_list))
    
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white")
    wordcloud.generate_from_frequencies(frequencies=words_list[i])
    
#     plt.figure()
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
    wordcloud.to_file("img/{}_word_clouds_{}_{}.png".format(age_grp,term_list[i], num_list))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




