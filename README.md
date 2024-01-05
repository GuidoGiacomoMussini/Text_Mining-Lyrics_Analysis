# Text_Mining MLM and Topic detection
Text mining course project.

The initial aim of this work was to 'extract the style' of italian songwriters and rewrite famous songs following that specific writing style. 

The major challenges for this task were: 
* The lack of data. In fact, the considered songwriters composed about 120 songs during their careers, offering different styles and topics.
* The complexity and numerosity of the vocabulary used by these authors, in relation to the number of songs produced.
* The lack of models, pre-trained specifically on the italian language, on which perform fine-tuning for the generation of decent lyrics, given the previous point.

A brief explanation of all the files in this repository follows.

## 1-Lyrics retrieval
This script exploit [the Genius API](https://genius.com/api-clients) to get the Lyrics. The data consists of the 120 most popular songs by 4 artists: Fabrizio De André, Giorgio Gaber, Francesco Guccini, Claudio Lolli. Free subscription to this API allows a limited number of monthly requests, but it is sufficient to get the data.

The resulting dataframe contains 3 columns: Author, Title and Lyrics. 

## 2-Preprocessing
The functions related to this script are in the preprocessing.py file.

This script is divided in 3 sections.   
The first one is related to the Lyrics-preprocessing, the second one to the Bible preprocessing and the last one to the 'Divina commedia' preprocessing.

*One of my first idea was to turn this problem into a 'translation' problem, a task excellently performed by transformers. The project, in a nutshell, was to create 2 models:   
The first one with the task of learning the writing style of a given author by analyzing his lyrics POS (part of speech) using an RNN or markov chain.   
The second one, a transformer model capable of translating a sequence of POS into written text. This second model would have been pretrained using Bible verses and then specialized on the lyrics of a second author.   
Thus, the idea was to use the syntax of one author and the semantics of the other. Unfortunately the models provided very poor results.*

#### Lyrics preprocessing
First of all, the lyrics obtained by Genius contained comments (i.e the phrase '*you might also like*' in the middle of almost each song) and headers that had to be removed. Here an example: 

*"10 ContributorsCoda di Lupo Lyrics[Testo di "Coda di lupo"] [Strofa 1] Quando ero piccolo m'innamoravo di tutto, correvo dietro ai cani E da marzo a febbraio mio nonno vegliava Sulla corrente di cavalli e di buoi Sui fatti miei e sui fatti tuoi E al dio degli inglesi non credere mai [Strofa 2] [...] You might also like[Strofa 5] Poi tornammo in Brianza per l'apertura della caccia al bisonte [...] e a un dio E a un dio E a un dio senza fiato non credere maiEmbed"*

After the processing:  
*"Quando ero piccolo m\'innamoravo di tutto, correvo dietro ai cani/  E da marzo a febbraio mio nonno vegliava/  Sulla corrente di cavalli e di buoi/  Sui fatti miei e sui fatti tuoi/  [...]  Poi tornammo in  Brianza per l\'apertura della caccia al bisonte/ [...] E a un dio senza fiato non credere mai"*

#### Outliers
The next step was to remove songs containing 'outlier verses' which are nothing more than pieces of prose within songs (common in Gaber). This procedure removed around 40 songs.

#### Text Preprocessing
In this section i've removed punctuation, uppercase, non-ASCII characters, and songs written in a foreign language or dialect (common in De André). This last step was not entirely successful; All the methods attempted failed expecially in deleting dialet songs. In the end, the library that provided the best results was [langdetect](https://pypi.org/project/langdetect/).

After these steps, i obtained:   
*"quando ero piccolo m'innamoravo di tutto correvo dietro ai cani/ e da marzo a febbraio mio nonno vegliava/ sulla corrente di cavalli e di buoi/ sui fatti miei e sui fatti tuoi"*,
where '/' is the symbol used to divide the lyrics by verses.

To conclude this part, i've created another column in the dataset, containing the lyrics after the stopwords removal (with [NLTK](https://www.nltk.org/search.html?q=stopwords)) and the text lemmatization (with [Spacy](https://spacy.io/api/lemmatizer)). This column is used in the 'Topic Detection' Section.

#### Divina Commedia Preprocessing. 
The data is taken by (http://it.wikisource.org). 

This text was used to train a classification algorithm to recognize whether two words rhyme with each other. The Divina Commedia was chosen because it is a very extensive text, about 14,000 verses, and the rhyme structure is well known.  
In this part, the text has been cleaned as the lyrics: comments, punctuation and non-ASCII symbols were removed.  
Then, the dataset used to train the classification algorithm was built: it has 2 columns, the first one contains the last word of each verse, the other one contains a numerical indication for the rhyme strucutre: ABA BCB CDC -> 010 121 232.  
Further details in section 4.

## 3-Topic Detection
The functions related to this script are in the topic_detection.py file.

In this section, I have tried to cluster the songs with respect to the main topic covered. Initially, the idea was to further specialize the model to generate a song according to both an author's style and the topic covered.    
The most common method to perform this taks, is the Latent Dirichlet Allocation (LDA), however, in this particular case this solution did not produce significant results. Moreover I didn't have any labels regarding the topics, so I couldn't use any supervised algorithm.  

To overcome this problem, I defined my own topic clustering algorithm. The algorithm is based on the concept of **FastText similarity**:  
[FastText](https://fasttext.cc/) is a library developed by facebook, which contains useful tools for NLP. In particular, I exploited its pre-trained Italian word embedding model to derive the absolute value of the cosine similarity between the vector representation of two given words: $S(x,y)$ indicates this similarity measure, where x,y are the vectorial representation of 2 words.  
I took the absolute value to bound the metric in [0,1], since for this purpose it is more important to find out *how strongly* two words are related, rather than *in what way* they are related.

This algorithm requires the definition of the topics to which the texts are assigned. I used single words to define a topic, but it is easy to generalize the method to accept a list of words to better represent a topic.  
The topics that i've defined for these data are: 'amore', 'dio', 'natura', 'politica', 'morte', 'guerra'.    
Since the similarity measure takes 2 words as inputs, you have to decide which words of each document (lyrics) have to be choosen to calculate it. The words that best represent the topics covered in a text are the NOUNs, which were extracted from each lemmatized text. One can choose to include other part of speech like ADJs or VERBs.   
In the final version of the script i used only the 10 most common NOUNs in each song, but the results are very similar.

#### FastText Classification Algorithm.

Assuming $A$ and $T$ are the list of words in a document and the list of topics, where each element in $A$ is denoted as $a_i$ and each element in $T$ is denoted as $t_j$.  
Let $S(x, y)$ be FastText similarity between the vectorial representation of the words $x$ and $y$.

The algorithm involves calculating the similarity between each word $a_i$ in $A$ and each topic $t_j$ in $T$. The document is assigned to the topic $t_k$ such that the sum of similarities between $t_k$ and all words in $A$ is greater than the sum of similarities between $t_j$ and all words in $A$, where $t_j$ belongs to $T \setminus \{ t_k \}$.

Mathematically, this can be expressed as:

$$t_k = \underset{t_j \in T}{\arg \max} \sum_{i=1}^{|A|} S(a_i, t_j)w_j $$

This notation signifies that we find the topic $t_k$ that maximizes the sum of similarities between $t_k$ and all words in $A$, ensuring it is greater than the sum of similarities between all the other topics $t_j$ and all words in $A$ for any $t_j$ in $T$.

The term $w_j$ $\in [0,1]$ refer to a metric called **Inverse popularity**:  
Popularity of a word represents how common the word is within the language.   
Let $X$ represent a random variable that randomly pick a word from a language, $c$ a common word and $n$ a less common word, where *common* refers to the word frequency (where frequency here indicates even the number of different meanings and context in which the word is used) in the dataset used to train the embedding model.  
Under the assumption that the words distribution in the datset is consisent with the words distribution in the language, we have that $P(S(c,x) > S(n,x))$ is probable for each x $\in$ X even if we can't directly recognize semantic or lexical link between $c$ and $x$.    
In other words, S(x,y) inherently tends to calculate a higher similarity score when a common word is involved. This bias leads the algorithm to frequently assign documents to topics represented by the most common term. 

To counteract this effect, the algorithm derives the *inverse popularity* coefficient $w_t$ for each topic t in T, defined as: $$w_t ∝  (\sum_{i=1}^{|L|} S(l_i, t_t))^{-1}$$ 
Here, $L$ is the list of each chosen word $l$ across all documents. This coefficeint is in fact proportional to the inverse of how 'popular' is the topic in the set of documents. Therefore, the most *popular* topics are penalized in the algorithm.  
A better and more general way to calculate this metric would be to derive it from the entire corpus of documents used to train the embedding model, as deriving it from the dataset on which the algorithm will be used will generate some bias in the coefficients.  
Note that each $w_t$ is in percentage, based on the sum of all the $w_t$ in T. It follows that as |T| increases, each $w_t$ decreases.  


For example, the topics popularity in my lyrics is:
 | |
|---|
|'amore': 0.197,
 'morte': 0.170,
 'ntura': 0.166,
 'politica': 0.162,
 'guerra': 0.155,
 'dio': 0.150|
 
An interesting aspect of this procedure is that it allows the extraction of the 'percentage belonging' of a text to each topic. This shows how much a text is related to a single topic or whether it range between different topics. For example, the topic distribution for the song 'Il testamento di Tito' is: 
 | |
|---|
|'dio': 0.190,
 'morte': 0.177,
 'amore': 0.167,
 'guerra': 0.162,
 'natura': 0.161,
 'politica': 0.143|


#### Song Similarity
Theoretically, the same logic can be adapted to find the most similar songs:    
Here, it is important to consider the same number of words for each song: the algorithm is not designed to weigh the lengths of the input lists, therefore songs with more words would be advantaged in the calculation, as the final score is the result of a simple sum between the similarities of each possible word pair of 2 songs. 

One aspect that deserves attention in this context is the impossibility of creating popularity weights. Since the 'topics' are every possible word in the whole set of songs, the popularity coefficient of each word would be flattened to zero by the cardinality of the set, making the inverse popularity-weights to tend to infinity by contrast. 

In this regard, it may be interesting to read the results obtained (On the script).  
It can be observed that the most similar songs were identified in pairs in which there is at least one love song (i.e. la stagione del tuo amore). This result is consistent with the previous findings, since 'amore' is the topic associated with the highest pupularity score.  
An interesting result is the third place (Preghiera in gennaio - Cristoforo Colombo), since these 2 songs don't have any specific reference to the 'love' area.

The lasts position are occupied by short songs or dialect songs (for which, probably, Spacy can't tag correctly the NOUNs), showing the balancing problem.

It might be interesting to try to adapt the method to handle the similarity calculation more precisely, perhaps trying to find an alternative way in defining the weights.

## 4-Rhyme Model
This script trains a model that can recognize whether 2 words rhyme. 

There are no specialized libraries to perform this task, excpecially for the italian language. One can adapt libraries such as NLTK or pyphen, or use some methods based on transforming words into their phonetic representation, but still they are not accurate. For this reason I preferred to build my own model. 

The model is not perfect:
* It makes errors -it reached only 90% accuracy on the test set-
* It is trained only on the Divina Commedia, which although it may be a good starting point, is definitely not enough to capture all the nuances and possible cases, especially in a language as complex as Italian.

The script imports the dataset defined in the 'preprocessing' section, in which each row contains 3 (due to the rhyme structure) words that rhyme.  
For each word trio, i've built 6 new rows in the training dataset, containing the 6 possible combinations of pairs (w1,w2,w3 -> w1-w2, w1-w3, w2-w3 and the reversed pairs).   
I added to the training dataset 1.5*len(training_dataset) example of words that don't rhyme (Note that the extraction was random, so it is possible that false negatives were generated at this stage).  
The training dataset then contains the column 'label', which is a binary variable such that is 0 if the two words rhyme, 1 otherwise.  
In conclusion of this part, I considered only the last 3 letters of each words. An example of row is: ('ito' - 'iro' - 0) or ('are'-'iso' - 1). 

The model used to perform this classification task is a feedforward neural network with a sigmoid activation function in the last layer. Two words were considered to rhyme if the output of this model is < 0.5.

For the future it may be interesting to develop this aspect by enriching the dataset and improving the model.

## 5 Masked Language Models
The functions related to this script are in the MLM_Utils.py files

*As mentioned above, the initial objective of this work was to try to generate lyrics following the style of a certain author, in order to try to rewrite a song by author A, following the style of songwriter B.  
In the first part of this discussion i've altready mentioned the 'POS to TEXT' translation model that wasn't able to provide any significant result.  
Another model that i've tried to implemet follows a similar logic: the idea was to train a trasformer to 'translate' a lemmatized (and after stopwords removal) text to a normal text, for example: 'binario stare locomotiva' -> 'e sul binario stava la locomotiva', trying to capture the way in which a given author composes the phrases, but i did not have sufficient examples to obtain meaningful results.*

I did not use data augumentaion techniques, since the structure of a song is extremely specific, changing one part of it could distort the structure.

The solution I found uses a Masked Language Model(MLM). It did not achieved satisfactory results in absolute terms, but they are the best I was able to obtain from these data. 

A Masked laungage model takes as input a string 'evaporato in una nuvola rossa' where one of the word is masked by a special token -> 'evaporato in una [MASK] rossa'. The task of this class of models is to predict the word masked by [MASK]. 'nebbia', 'bottiglia', 'nuvola', 'notte', for example.  
My idea is to sequentially use an MLM on a verse, to iteratively modify it according to the replacements suggested by a fine-tuned MLM model.

#### Training
I used the [BERT for MLM -'bert-base-italian-xxl-cased'](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased) model as basis.  
Then i've selected and extracted all the verses of all of the songs by a given author.   
I worked for the most at verse-level since i didn't have evidence of the strophe for all the lyrics and it was computationally costly to work at lyric-level: I trained a model directly on the lyrics, but i had to impose the encoding limit to 100 tokens and it provided very poor results.

The dataset was built by sequentially replacing each word of each verse with the special token, for example: 
 | |
|---|
|'[MASK] in una nuvola rossa'
 'evaporato [MASK] una nuvola rossa'
 'evaporato in [MASK] nuvola rossa'
 'evaporato in una [MASK] rossa'
 'evaporato in una nuvola [MASK]'|

The label associated to these inputs was the complete verse 'evaporato in una nuvola rossa'.

The dataset has been used to fine-tune the MLM BERT model. One can even filter the data by the topic defined before, but this will reduce the cardinality of the set.

#### Generation 
The generation of text is an iterative process, i've defined a base method and a more complex one.  
Assume a verse of an author (ideally, a different author drom the one used in the fine-tuning), i.e: 'ma se io avessi previsto tutto questo'.

**base method** : 

1. Select a song
2. The first step is to mask the first element: phrase_1 = '[MASK] se io avessi previsto tutto questo'
3. phrase_1 is the input of the MLM BERT model, which returns, for example: phrase_2 ='e se io avessi previsto tutto questo' 
4. Then we mask the second element of the output of MLM BERT, phrase_2: 'e [MASK] io avessi previsto tutto questo' 
5. The process is repeated iteratively for each element in the verse
6. Repeat the process for each verse in the song

Regarding the point 3: an interesting feature of MLM_BERT is that it allows you to extract the dictionary words-probabilities taken into account to replace the [MASK]. I have exploited this feature to bound the choice of the word, to the one with the higher probability of the same POS of the masked element:   
'evaporato in una [MASK] rossa', since 'nuvola' is a NOUN, the model will select from the dictionary the NOUN with the higher probability.  
This could lead to problems since the POS is determinated even by the context (i.e 'sale' could be NOUN or VERB), but it is useful to provide more 'structure' 
 to the process. 

In general, this procedure has the advantage of exploiting MLM_BERT to iteratively fill the masked words, in doing so, at each step it manages to capture the context of the input-verse and modify it following what it has learned during the fine tuning. On the other hand, it is not capable to manage the rhymes structure, since it is trained on verse-level.

**advanced method**:  

In order to try to overcome the rhyme problem, i've used the Rhyme Model defined above to extract the rhyme structure of the chosen song.   
I've used the rhyme structure to bound the exctraction of the last word of each verse to a word that rhyme accordingly to the structure. Unfortunately, the number of 'meaningful' possible replacement is limited and it is difficult to find a word with the desired characteristics.

Here 2 examples of results provided by the simple and complex method, using the simple MLM BERT Model.  
(Original song: Un matto - Fabrizio De André)
 | |
|---|
| [...] |
|Ma per sapere a chi spetta la pensione
In un mese te ne fa
Qui in città parecchi anni
Non ce traccia nemmeno nei miei occhi
Ora in scena non trovate parole
E ha la luce la luce del sole| 
| [...] |

 | |
|---|
| [...] |
|Ma per sapere a chi spetta la responsabilità
In un mese te ne farò
Qui in città parecchi stranieri
Non ce traccia nemmeno nei miei pensieri
Ora in scena non trovate parole
E ha la luce la luce del sole|
| [...] |

It can be observed that the two texts are similar. Many verses have internal logic but obviously lack textual consistency. The use of the rhyme structure worked in the third and fourth verses of the second table, which are different from their counterparts in the first one. (the original verses are: 'qui sulla collina dormo malvolentieri / eppure c'è luce ormai nei nei miei pensieri')

Another example, using a model fine-tuned on De André (original song: Cyrano-Francesco Guccini)

 | |
|---|
| [...] |
|le anime lasciate sulla strada dai cani
lascio le mani muove le dita
andate a caval leva veloce indietro
per la tua barba lunga ti sembrano giganti
ai dubbi e ai sogni di domani non credete
e in fin di vita io poi raccol e pe
io non chiedo mai perdono ma amore
io cerch i miei occhi col coltello e con la pietra
ma in questa città dove non passa più la notte
non so provare ad essere felice| 
| [...] |


 | |
|---|
| [...] |
|le anime lasciate sulla strada dai criminali
lascio le mani muove le ali
andate a caval leva veloce indietro
per la tua barba lunga ti sembrano pietà
ai dubbi e ai sogni di domani non credete
e in fin di vita io poi raccol e m
io non chiedo mai perdono ma amore
io cerch i miei occhi col coltello e con la cu
ma in questa città dove non passa più la notte
non so provare ad essere cattivo| 
| [...] |

Here you can see that the advanced model was able to replicate the rhyme schema (A,A,B,B,C,C,C,D,D,E..) only for the first 2 verses.   
Moreover there is some word that probably is misinterpreted by the tokenizer: 'raccol e m', 'e con la cu'. On the other hand there are meaningful verses: 
*'ma in questa città dove non passa più la notte / non so provare ad essere cattivo* or *io non chiedo mai perdono ma amore* 

Last example: A comparison of fine-tuned simple model on Guccini and fine-tuned simple model on De André (original song: Illogica Allegria - Gaber).  

Guccini 'traslation':

 | |
|---|
| [...] |
|da lontano
lungo l
alle chiaro luci del giorno
a volte fa male la testa
e senti il tuo viso passare sopra ride
io parlava di canzoni e molto del vino
lo senti che ti senti in mente
ma a volte
mentre la notte cade
al suo strano profumo| 
| [...] |

De André 'traslation':

 | |
|---|
| [...] |
|da solo
lungo l
alle prime ore del mattino
a volte suona ancora la chitarra
e vedo il mio cuore venire dal pianto
ne va del mondo e poi del mondo
lo sai che si muore in battaglia
ma di dolore
mentre la regina piange
del tuo profondo dolore| 
| [...] |

It seems that the De André model perform better since it changes completely some verses ('a volte spengo anche la radio'-> 'a volte suona ancora la chitarra'), in a way that can (distantly) resemble his Style.
 






