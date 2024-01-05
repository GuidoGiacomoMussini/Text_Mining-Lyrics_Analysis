# DEPENDENCIES
import subprocess

libs = ["spacy", "fasttext"]
for lib in libs:
    subprocess.run(["pip", "install", lib])

from collections import Counter
import numpy as np
import spacy
import fasttext
import fasttext.util

subprocess.run(["python", "-m", "spacy", "download", "it_core_news_sm"])

nlp_spacy = spacy.load("it_core_news_sm")
fasttext.util.download_model('it', if_exists='ignore')  #model download
similarity_model = fasttext.load_model('cc.it.300.bin') #pretrained model


# FUNCTIONS
def extract_POS(text, POS):
  '''
  exctract all the POS(part of sspeech) in a given list in a text (i.e. extract all the NOUNS )
  '''
  doc = nlp_spacy(text)
  nouns = [token.text for token in doc if token.pos_ in POS]

  return nouns


def convert_in_POS(text):
  '''
  Convert a text in a text of POS
  '''
  doc = nlp_spacy(text)
  pos_text = [token.pos_ for token in doc]

  return pos_text


def common_words(words, top):
  '''
  extract the most common words in a list
  '''
  word_counts = Counter(words)
  most_common_words = word_counts.most_common(top)
  return [word for word, count in most_common_words]


def cosine_similarity(a, b): 
  vector_a = similarity_model.get_word_vector(a)
  vector_b = similarity_model.get_word_vector(b)
  return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


def percentage_weight(dict_):
  '''
  convert the values of a dict in percentages of sum(values)
  '''
  tot_weight = sum(dict_.values())
  topics_weight = {key: (round(value / tot_weight, 3)) for key, value in dict_.items()}

  return topics_weight


def sort_dict(dict_, reverse = True):
  '''
  sort a dict on values
  '''
  return dict(sorted(dict_.items(), key=lambda item: item[1], reverse=reverse))


def find_topic_similarity(topics, word_list, weight = False, normalize = True):

  '''
  sum the similaritis between each noun in a song and topic.
  If weight = True, the similarity is weighted by a measure that penalize
  the topics that have stronger similarities with all the words in the df.
  '''
  similarity_dict = {}

  for topic in topics:
    #define the weights
    if weight: w = topics[topic]
    else: w = 1

    #initialize the similarity measure
    similarity = 0

    #derive the sum of the similarity between each word in a song and a given topic
    for word in word_list:
      
      similarity += np.abs(cosine_similarity(word, topic))/(w)

    similarity_dict[topic] = similarity #feed the dict

  if normalize == True: #trasform in percentages

    perc_similarity_dict = percentage_weight(sort_dict(similarity_dict)) #sort the dict and trasform the value in percentage
    
    return perc_similarity_dict
  
  else: 
    return similarity_dict


def find_popularity(topics, vocabulary):
  #unique words  --> you can choose to count each word in the set only once, but that will further penalize the common words
  #vocabulary = list(set(vocabulary))

  #find similarity btwn each topic and each word in the song
  vocabulary_topic_similarity = find_topic_similarity(topics, vocabulary)

  #trasform the weight in percentage
  topics_weight = percentage_weight(vocabulary_topic_similarity)

  return topics_weight


def song_similarity(song1, song2):
  '''
  derive the similarity between 2 songs
  '''
  similarity_dict = find_topic_similarity(song1, song2, weight = False, normalize = False)

  return sum(similarity_dict.values())
