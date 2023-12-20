import subprocess
subprocess.run(["pip", "install", 'spacy'])
import spacy
subprocess.run(["python", "-m", "spacy", "download", "it_core_news_sm"])


import numpy as np
import torch
import random
from torch.utils.data import Dataset
nlp = spacy.load("it_core_news_sm")


# RHYMES
def word_to_features(word, num_features):
  '''
  create numerical index to indicate the last n (num features) letters
  '''
  #extract the lasts n letters, if a word is shorter, than apply padding value
  padding_value = 0
  features = [ord(char) for char in word[-num_features:]] if len(word) >= num_features else [padding_value] * (num_features - len(word)) + [ord(char) for char in word]

  return features



#TRAINING
def mask_all_string(string_):
  '''
  Each word of a text-verse e is sequentially replaced by [MASK]. 
  Each resulting masked-string is paired with the original text-verse
  '''
  v_list = string_.split()
  masked_list = []
  string_list = []
  for i in range(len(v_list)):
    masked_list.append(' '.join(['[MASK]' if j == i else v_list[j] for j in range(len(v_list))]))
    string_list.append(string_)

  return masked_list, string_list


def create_masked_pairs(list_verses):
  '''
  apply 'mask_all_string' to all the verses in the data, creating the dataset
  '''

  x, y = [], []
  for verse in list_verses:

    mask_, string_ = mask_all_string(verse)
    x += mask_
    y += string_

  return x, y


class BertDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = df['x'].tolist()
        self.labels = df['y'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        labels = self.labels[idx]
      
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        label_encoding = self.tokenizer.encode_plus(
            labels,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_encoding['input_ids'].flatten()
        }
    

# GENERATION
 
def get_pos(word):
  '''
  get the POS of a word. Since the POS could depend on the context ('sale' as verb or noun)
  this is only an approximation
  '''
  doc = nlp(word)
  return doc[0].pos_ if doc else None


def token2text(model, tokenizer, token_list, mask_token_logits):
  '''
  translate the outputs from tokens to text
  
  '''
  word_prob = {}
  model.eval()

  for token in token_list:
    word = tokenizer.decode([token])
    prob = torch.softmax(mask_token_logits, dim=1)[0, token].item()

    word_prob[word] = round(prob, 5)

  return word_prob


def extract_output_dict(model, tokenizer, verse, k):
    '''
    extract the dictionary of the first k guess of the model for the [MASK]

    '''
    try:
        inputs = tokenizer.encode(verse, return_tensors="pt")
        mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.logits

        mask_token_logits = predictions[0, mask_token_index, :]
        top_k_tokens = torch.topk(mask_token_logits, k, dim=1).indices[0].tolist()
        ris = token2text(model, tokenizer, top_k_tokens, mask_token_logits)
    except:
        ris = {'UNK': 1}

    return ris


def modify_verse(model, tokenizer, verse, num_extractions):
    '''
    modify sequentially a verse. The mask is put in place of the first word and predicted,
    then the mask is put in place of the second element of the resulting string and so on.
    '''
    v_list = verse.split()
    for i in range(len(v_list)):
        original_word_pos = get_pos(v_list[i])
        masked_verse = v_list.copy()
        masked_verse[i] = '[MASK]'
        v_string = ' '.join(masked_verse)

        predictions = extract_output_dict(model, tokenizer, v_string, num_extractions)
        filtered_predictions = {word: prob for word, prob in predictions.items() if get_pos(word) == original_word_pos}

        #if not valid predictions -> use bert best one
        if filtered_predictions:
            sub = max(filtered_predictions, key=filtered_predictions.get)
        else:
            sub = v_list[i]

        v_list[i] = sub

    return ' '.join(v_list)


def check_rhyme(word1, word2, model, num_features):
  '''
  It takes 2 words (strings) and checks whether they rhyme.
  '''
  #extract the last n letters from the words
  features_word1 = word_to_features(word1, num_features)
  features_word2 = word_to_features(word2, num_features)

  #adjust the dimension for the model
  features_word1 = np.expand_dims(features_word1, axis=0)
  features_word2 = np.expand_dims(features_word2, axis=0)

  #classification
  prediction = model.predict([features_word1, features_word2], verbose = 0)

  return prediction[0][0] < 0.5



def find_rhyme_scheme(word_list, model, num_features):
  '''
  the input is the list of the last word of each verse in a song. 
  the output is a list representing the rhyme schema: [0,0,1,1] --> A,A,B,B
  '''
    
  rhyme_scheme = []
  rhyme_index = 0
  rhyme_groups = {}
  k = 0
  for i, word1 in enumerate(word_list):
      if i not in rhyme_groups:
          rhyme_groups[i] = rhyme_index
          rhyme_index += 1
      
      for j, word2 in enumerate(word_list[i+1:], start=i+1):
          if j not in rhyme_groups and check_rhyme(word1, word2, model, num_features):
              rhyme_groups[j] = rhyme_groups[i]

  for i in range(len(word_list)):
      rhyme_scheme.append(rhyme_groups.get(i, -1))

  return rhyme_scheme


def check_rhyme_with_scheme(modified_verse, normalized_rhyme_scheme, verse_index, song, rhyme_model, num_features):
  '''
  Checks if the last word of a modified verse rhymes with corresponding words in other verses of a song.
  For each verse in the song, it checks if it is supposed to rhyme with the current verse. 
  If so, it uses the rhyme_model to determine if the last word of the current verse rhymes with the last word of the other verse. 

  '''
  last_word = modified_verse[-1]
  for other_verse_index, rhyme_group in enumerate(normalized_rhyme_scheme):
      if rhyme_group == normalized_rhyme_scheme[verse_index] and other_verse_index != verse_index:
          other_last_word = song[other_verse_index].split()[-1]
          if not check_rhyme(last_word, other_last_word, rhyme_model, num_features):
              return False
  return True


def modify_verse_with_rhyme(BERT_model, rhyme_model, tokenizer, verse, num_extractions, normalized_rhyme_scheme, verse_index, song, num_features):
  '''
  Modifies a verse to maintain or create consistent rhymes with other verses in a song.
  The function iterates over each word in the verse. 
  For each word, it creates a copy of the verse with the current word replaced by a '[MASK]' token and generates a set of predictions for the masked word using the BERT model. 
  It then filters these predictions based on the part of speech of the original word. 
  For the last word in the verse, it tries to find a predicted word that maintains rhyme with other verses as defined in the normalized_rhyme_scheme. 
  If no valid rhyme is found, it uses the word with the highest probability that matches the original part of speech. For non-final words in the verse, 
  it selects the word with the highest probability that matches the original part of speech.
  '''
  v_list = verse.split()
  for i in range(len(v_list)):
      original_word_pos = get_pos(v_list[i])
      masked_verse = v_list.copy()
      masked_verse[i] = '[MASK]'
      v_string = ' '.join(masked_verse)

      predictions = extract_output_dict(BERT_model, tokenizer, v_string, num_extractions)
      filtered_predictions = {word: prob for word, prob in predictions.items() if get_pos(word) == original_word_pos}

      if i == len(v_list) - 1:
          rhyme_found = False
          for predicted_word in filtered_predictions.keys():
              modified_verse = v_list.copy()
              modified_verse[i] = predicted_word
            #if rhyme is found, use the word
              if check_rhyme_with_scheme(modified_verse, normalized_rhyme_scheme, verse_index, song, rhyme_model, num_features):
                  v_list[i] = predicted_word
                  rhyme_found = True
                  break
          
          # If no rhyme, use the word with higher probability
          if not rhyme_found:
              v_list[i] = max(filtered_predictions, key=filtered_predictions.get, default=v_list[i])
      else:
          # For the other words, use the POS logic
          v_list[i] = max(filtered_predictions, key=filtered_predictions.get, default=v_list[i])

  return ' '.join(v_list)


def modify_song(song, BERT_model, tokenizer, num_extractions, rhyme_model, num_features):

  #extract last word of each verse
  last_words = [verse.split()[-1] for verse in song]

  #derive the ryhmes schema
  print("deriving the rhyme schema")
  rhyme_scheme = find_rhyme_scheme(last_words, rhyme_model, num_features)
  print("-"*15)


  print("modify verse by verse the song:\n")
  #generate text as described
  modified_song = []
  for verse_index, verse in enumerate(song):
      modified_verse =modify_verse_with_rhyme(BERT_model, rhyme_model, tokenizer, verse, num_extractions, rhyme_scheme, verse_index, song, num_features)
      modified_song.append(modified_verse)
      print(modified_verse)

  return modified_song
