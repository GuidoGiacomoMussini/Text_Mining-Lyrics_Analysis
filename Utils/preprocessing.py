# DEPENDENCIES
import subprocess

libs = ["nltk", "unidecode", "langdetect"] #"stanza"
for lib in libs:
    subprocess.run(["pip", "install", lib])

import re
from unidecode import unidecode
import unicodedata
from langdetect import detect
import spacy
import nltk
from nltk.corpus import stopwords
#import stanza

subprocess.run(["python", "-m", "spacy", "download", "it_core_news_sm"])
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#nlp_stanza = stanza.Pipeline("it")
nlp_spacy = spacy.load("it_core_news_sm")



# FUNCTIONS

def divide_uppercase(text):
    '''
    The data source added to a large part of the lyrics 'wordYou might also like..'. 
    This function separate the words with an uppercase letter in the middle: "exAmple"-> "ex Ample"
    '''
    divided_text = ""
    for char in text:
        if not char.isupper():
            divided_text += char
        else:
            divided_text += " " + char
    return divided_text


def remove_headers(text):
  '''
  remove the headers from the text.
  Each song start with somethingLyrics... and end with ...Embed 
  Since the first header has not the same size in each song, the function looks for the word 'lyrics'
  'Embed' is instead constant in each song, so the function simply remove the lasts characters in the string
  '''
  header = "Lyrics"
  lyrics_index = text.find(header) #lyric index
  no_header_text = text[lyrics_index + len(header):] #remove the header

  return  no_header_text.lstrip()[:-5] #remove 'Embed'


def clean_lyrics(raw_text):
  '''
  remove the non-lyric text from the lyrics and modify the way in which verses are indicated
  '''
  #remove the text in '[..]' (i.e [intro[, [strofa1], [ritornello] etc)
  text = re.sub(r'\[[^\]]*\]', '', raw_text)
  #remove the '/' after the apostrophes
  text = text.replace("\\'", "'")

  #replace '\n' with '/ ' to separate the verses
  text = text.replace("\n", "/ ")
  #cleaned_lyrics = re.sub(r'/+', '/', cleaned_lyrics)

  #separate the uppercase and delete the phrase 'You might also like'
  text = divide_uppercase(text).replace('You might also like', '')

  #remove the empty verses (deleteing [..] we had some '\ \ \' in the text)
  text_list = [verse + "/" for verse in text.split("/") if verse.strip()]

  #list -> string
  text = "".join(text_list)[:-1]

  return text.strip()


def delete_chars(text, del_char):
  '''
  remove all the characters in the string 'del_char' from the string 'text'
  '''
  for char in del_char:
    text  = text.replace(char, '')
    
  return text


def remove_punctuation(text, punctuation_pattern):
  '''
  remove the punctuation in the string 'punctuation_patterns' from the string 'text'
  '''
  text = re.sub(punctuation_pattern, ' ', text)

  return text


def fix_accents(text, accepted_accents):
  '''
  standardize the accents in the string 'text'
  '''
  result = [char if char in accepted_accents else unidecode(char) for char in text]

  return ''.join(result)

def text_language(text):
   '''
   the function 'detect' return the language of the 'text' string, but it accepts a string in the form 'word1, word2, word3, ..'
   '''
   return detect(", ".join(text.split(" ")))

def remove_stopwords(text):
  '''
  remove all the stopwords from the string 'text'
  '''
  stop_words = set(stopwords.words('italian')) #nltk function
  words = text.split()
  filtered_words = [word for word in words if word.lower() not in stop_words]

  return ' '.join(filtered_words)


#def lemmatization_stanza(text_string):
#  '''
#  perform lemmatization in a text using 'stanza' (good results with the italian language but computationaly expensive)
#  '''
#  doc = nlp_stanza(text_string)
#  lemmi = [word.lemma for sentence in doc.sentences for word in sentence.words]
#  lemmi_string = ' '.join(lemmi)
#  return lemmi_string


def lemmatization_spacy(text_string):
  '''
  perform lemmatization in a text using 'spacy' (some error with the italian language but computationaly fast)
  '''
  doc = nlp_spacy(text_string)
  lemmi = [token.lemma_ for token in doc]
  lemmi_string = ' '.join(lemmi)

  return lemmi_string


def rhymes_progression_idx(number_of_verses): 
  '''
  create the rhymes schema used in Divina Commedia 'ABA BCB CDC..' -> '121 232 343..'
  '''
  progression =  []
  iter = 1
  for i in range(number_of_verses): 
    progression.append(iter)
    progression.append(iter+1)
    progression.append(iter)
    iter +=1
  else: 
    progression.append(iter)
  return progression


def pos_tagging(text):
  '''
  from text to part of speech
  '''
  doc = nlp_spacy(text)
  list_ = [token.pos_ for token in doc]
  return list_
