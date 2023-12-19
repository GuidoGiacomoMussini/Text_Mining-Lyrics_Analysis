# Text_Mining-Lyrics_Analysis
Text mining course project. 

The initial aim of this work was to generate lyrics 'extract the style' of italian songwriters and rewrite famous songs following that specific writing style. 
The major challenges for this task were: 
* the scarcity of data. In fact, the songwriters considered composed about 120 songs during their careers, offering different styles and topics from each other. 
* The lack of pre-trained models specifically on Italian on which to fine-tune and usable for the generation of good lyrics, given the previous point.

A brief explanation of all the files in this repository follows

## 1-Lyrics retrieval
This script exploit [the Genius API](https://genius.com/api-clients) to get the Lyrics data. The data consists of the 120 most popular songs of 4 artists: Fabrizio De André, Giorgio Gaber, Francesco Guccini, Claudio Lolli. 
free subscription to this API allows a limited number of monthly requests, but it is sufficient to get the data. 
The resulting dataframe containt 3 columns: Author, Title and Lyrics. 

## 2-Preprocessing
The functions related to this script are in the preprocessing.py file.
This script is divided in 3 sections. 
The first one is releted to the Lyrics-preprocessing, the second one to the Bible preprocessing and the last one to the 'Divina commedia' preprocessing.

*One of my first ideas to complete this task, was to turn this problem into a 'translation' problem, a task excellently done by transformers. The project, in a nutshell, was to create 2 models: the first would have the task of translating the lyrics of a given author into POS (part of speech) and using an RNN or markov chain to reproduce the author's writing style; The second a transformer model capable of translating a sequence of POS into a written text. This second model would be pretrained using Bible verses, then specialized on the style of a second author. Thus, the idea was to use the syntax of one author and the semantics of the other. Unfortunately the models provided very poor results. In any case, for those interested, the code for generating the two models is in the 'Extras' folder*

#### Lyrics preprocessing
First of all, song lyrics contained comments(like the phrase 'you might also like' in the middle of almost each song), header that have to be removed. Here is an example: 
*10 ContributorsCoda di Lupo Lyrics[Testo di "Coda di lupo"] [Strofa 1] Quando ero piccolo m'innamoravo di tutto, correvo dietro ai cani E da marzo a febbraio mio nonno vegliava Sulla corrente di cavalli e di buoi Sui fatti miei e sui fatti tuoi E al dio degli inglesi non credere mai [Strofa 2] [...] You might also like[Strofa 5] Poi tornammo in Brianza per l'apertura della caccia al bisonte [...] e a un dio E a un dio E a un dio senza fiato non credere maiEmbed*

After the processing: 
*Quando ero piccolo m\'innamoravo di tutto, correvo dietro ai cani/  E da marzo a febbraio mio nonno vegliava/  Sulla corrente di cavalli e di buoi/  Sui fatti miei e sui fatti tuoi/  [...]  Poi tornammo in  Brianza per l\'apertura della caccia al bisonte/ [...] E a un dio senza fiato non credere mai*

#### Outliers
The next step was to remove songs containing 'outlier verses' which are nothing more than pieces of prose within songs (common in Gaber). This procedure removed around 40 songs.

#### Text Preprocessing
In this section have been removed punctuation, capitalization, any special accents or symbols, and even songs written in a foreign language or dialect (common in De André). This step was not entirely successful; all the libraries tried failed to remove the entirety of the foreign language songs. In the end, I opted to use [langdetect](https://pypi.org/project/langdetect/)

After these processes, i obtained: *quando ero piccolo m'innamoravo di tutto correvo dietro ai cani/ e da marzo a febbraio mio nonno vegliava/ sulla corrente di cavalli e di buoi/ sui fatti miei e sui fatti tuoi*, where '/' is the symbol used to divide the verses.

In the end another column was created in the dataset, with the lyrics after the stopwords were removed and the text lemmatized. This is used in the 'Topic Detection' Section.

#### Divina Commedia Preprocessing. 
The data is taken by (http://it.wikisource.org). 
This text was used to train a classification algorithm that can recognize whether two words rhyme with each other. The Divina Commedia was chosen because it is a very extensive text, about 14,000 verses, and the rhyme structure is well known.
In this part, the text has been cleaned as the lyrics: comments, punctuation and non-ASCII symbols were removed. 
Then, the dataset used to train the classification algorithm was then built: it contains 2 columns, the first one contains the last word of each verse, the other column contains a numerical indication for the rhyme strucutre: ABA BCB CDC -> 010 121 232.

## 3-Topic Detection
The functions related to this script are in the topic_detection.py file.

In this section, i have tried to cluster the songs with respect to the main topic covered. initially, the idea was to further specialize the model, to train it to generate a song according to both an author's style and the topic covered.

The most common method of achieving this is the Latent Dirichlet Allocation (LDA), however, in this particular case this algorithm did not produce significant results, probably due to the type of documents. Moreover i don't have any labels regarding the topics, so i couldn't use any type of supervised algorithm. 
To overcome this problem, i defined my own topic classification algorithm. The algorithm is based on the concept of [FastText](https://fasttext.cc/) similarity. Fasttext is a library developed by facebook, which contains useful tools for NLP. In particular, i exploit its pre-trained Italian word embedding model to derive the cosine similarity between the vector representation of two given words: S(x,y) where x,y are the words.

#### FastText Classification Algorithm. 
This algorithm requires defining the topics to which the texts are to be assigned. I used single words to define a topic, but it is easy to generalize the method to accept a list of words representing a topic. 
I decided to define 6 topics: 'amore', 'dio', 'natura', 'politica', 'morte', 'guerra'.  
Since the similarity measure take as inputs 2 words, you have to decide which words of each document have to be choosen to calculate it. The words that best represent a text are the NOUNs, which were extracted from each lemmatized text, you can choose to include other part of speech like ADJs or VERBs. In the final version of the script i used only the 10 most common NOUNs in each text, but the results are very similar.

Assuming $A$ and $T$ are the list of words in a document and Topic, respectively. Where each element in $A$ is denoted as $ a_i$ and each element in $T$ is denoted as $t_j$.
Let $S(x, y)$ be the cosine similarity between the vectorial representation of the words $x$ and $y$.

The algorithm involves calculating the similarity between each word $a_i$ in $A$ and each topic $t_j$ in $T$. The document is assigned to the topic $t_k$ such that the sum of similarities between $t_k$ and all words in $A$ is greater than the sum of similarities between $t_j$ and all words in $A$, where $t_j$ belongs to $T \setminus \{ t_k \}$.

Mathematically, this can be expressed as:
$$ t_k = \underset{t_j \in T}{\arg \max} \sum_{i=1}^{|A|} S(a_i, t_j)w_j > \sum_{i=1}^{|A|} S(a_i, t_k)w_k $$
This notation signifies that we find the topic $t_k$ that maximizes the sum of similarities between $t_k$ and all words in $A$, ensuring it is greater than the sum of similarities between $t_j$ and all words in $A$ for any $t_j$ in $T \setminus \{ t_k \}$.

The terms $w_j$, $w_k$  $\in [0,1]$ refer to a metric that i've called **Self Similarity**:
Self-similarity of a word represents how common the word is within the language. 
Let $X$ represent a random variable that randomly pick a word from a language, $c$ a common word and $n$ a less common word, where *common* refers to the word frequency in the dataset used to train the embedding model. Under the assumption that the words distribution in the datset is consisent with the words distribution in the language, we have that $P(S(c,x) > S(n,x))$ is probable for each x $\in$ X while recognizing no direct semantic or lexical link between $c$ and $x$.  
In other words, S(x,y) inherently tends to calculate a higher similarity score when a common word is involved. This bias leads the algorithm to frequently assign documents to topics represented by the most common term. 

To counteract this effect, the algorithm derives *self-similarity* coefficient $w_t$ for each topic t in T, defined as: $$ w_t ∝  (\sum_{i=1}^{|L|} S(l_i, t_t))^{-1} $$ Here, $L$
is the list of each chosen word $l$ across all documents. *Note that each $w_t$ is taken in percentage, based on the sum of all the w_t in T. Tt follows that as T increases, each $w_t$ decreases.*
So this coefficeint is proportional to the inverse of how 'popular' is the word representing the topic in the set of documents.
For example, the popularity of topics in the lyrics of my songs is:
 | |
|---|
|'dio': 0.32,
 'amore': 0.186,
 'morte': 0.162,
 'guerra': 0.124,
 'natura': 0.123,
 'politica': 0.085|

The self similarity score is nothing but (popularity score)^{-1}. Therefore, the most *self-similar* topics are penalized in the algorithm.

An interesting aspect of this procedure is that it allows the extraction of the 'percentage belonging' of a text to each topic. This shows how much a text is related to a single topic or whether it spaces between different topics, for example, the topic distribution for the song 'Il testamento di Tito' is: 
 | |
|---|
|'dio': 0.206,
 'amore': 0.18,
 'morte': 0.176,
 'natura': 0.154,
 'guerra': 0.147,
 'politica': 0.136|

 Note that even if 'dio' is the most penalized topic, it were in any case assigned to the given song (and it is a perfectly justified assignment)


#### Song Similarity
The same logic can be adapted to find the most similar songs: 
In this case it is important to consider the same number of words for each song: since the algorithm is not designed to weigh the lengths of the input lists, songs with more words would be advantaged in the calculation, as the final score is the result of a simple sum between the similarities of each possible word pair of 2 songs. 

One aspect that deserves attention in this context is the impossibility of creating self-similarity weights. Since the 'topics' are every possible word in the whole set of songs, the popularity coefficient of each word would be flattened to zero by the cardinality of the set, making the self-similarity weights tend to infinity by contrast. 

In this regard, it may be interesting to read the results obtained (found on the script) 
It can be seen that the most similar songs were indentified in dialect songs, probably due to problems in the handling of such words by the embedding model. 
The other first places, however, are occupied by songs about faith or God, Topic which, before adding self-similarity, I have seen to be the one that generates the most bias in the similarity calculation.

It might be interesting to try to adapt the method to handle the similarity calculation more precisely, perhaps trying to find an alternative way in defining the weights


## 4-Rhyme Model
This script trains a model that can recognize whether 2 words rhyme. 
There are no specialized libraries to perform this task, one can adapt libraries such as NLTK or pyphen, or use some methods based on transforming words into their phonetic representation, but still they are not 100% accurate. For this reason I preferred to build a model optimized on the Italian language. 
The model is not perfect:
* It makes errors -it reached only 90% accuracy on the test set-
* It is trained only on the Divina Commedia, which although it may be a good starting point, is definitely not enough to capture all the nuances and possible cases, especially in a language as complex as Italian.

The script import the dataset defined in the 'preprocessing' section, in which each row contains 3 (due to the rhyme structure) words that rhymes.
For each word trio, i've built 6 new rows in the training dataset: containing the 6 possible combinations of pairs (w1,w2,w3 -> w1-w2, w1-w3, w2-w3 and the reversed pairs). 
I added to the training dataset 1.5*len(training_dataset) example of words that don't rhyme ( Note that the extraction was done randomly, so it is possible that false negatives were generated at this stage)
The training dataset contains then the label, which is a binary variable such that is 0 if the two words rhymes, 1 otherwise.
In conclusion of this part, i considered only the last 3 letters of each words in the training dataset. An example of row can therefore be: ('ito' - 'iro' - 0) or ('are'-'iso' - 1). 

The model used to perform this classification task is a feedforward neural network with a sigmoid activation function in the last layer. Two words were considered to rhyme if the output of this model is < 0.5.

for the future it may be interesting to develop this model by enriching the dataset and improving the model.

## 5 Masked Language Models
The initial objective of this work was to try to generate lyrics following the style of a certain author, to try to rewrite a song by author A, following the style of singer-songwriter B.
In the first part of this discussion i've altready mentioned the 'POS to TEXT' translation model that wasn't able to provide any significant result. 
Another model that i've tried to implemet follow a similar logic: the idea was to train a trasformer to 'translate' a lemmatized (and after stopwords removal) text to a normal text: for example: 'binario stare locomotiva' -> 'e sul binario stava la locomotiva'. The aim was to capture the way in which a given author composes the phrases, but again i didn't have sufficient examples to obtain meaningful results. 

I did not use data augmentation techniques to enrich the dataset as changing the lyrics of a song by, for example, using synonyms, can alter the structure of the lyrics.

The solution i found uses a Masked Language Model(MLM). It did not bring satisfactory results in absolute terms, but in any case they are the best I was able to obtain from these data. 
A Masked laungage model take as input a string 'evaporato in una nuvola rossa' where one of the word is masked by a special token [MASK] -> 'evaporato in una [MASK] rossa', the task of this class of models is to predict the word masked by [MASK]. 'nebbia', 'foschia', 'nuvola', 'notte', for example. 
my idea is to sequentially use an MLM on a sentence, to transform it according to what the model has learned. 

#### Training. 
I used the [BERT-'bert-base-italian-xxl-cased'](https://huggingface.co/dbmdz/bert-base-italian-xxl-cased) model as a basis. 
Then i selected one author, let's say De André, and i've extracted all the verses of all of his songs. 
I had to work on verse-level since i didn't have evidence of the strophe for all the lyrics, and work at lyric-level was computationally too costly.
I built the dataset by sequentially replacing each word of each verse with the special token, for example: 
 | |
|---|
|'[MASK] in una nuvola rossa'
 'evaporato [MASK] una nuvola rossa'
 xz'+sxcevaporato in [MASK] nuvola rossa'
 'evaporato in una [MASK] rossa'
 'evaporato in una nuvola [MASK]'|
the label of each of these input was the complete phrase 'evaporato in una nuvola rossa'
I then used this dataset to fine-tune the BERT model. 

One can even filter the data by the topic defined before.

#### Generation 
The generation of text is an iterative process, shortly:
Assume that we have a verse of another author: 'ma se io avessi previsto tutto questo'
The first step is to mask the first element: phrase_1 = '[MASK] se io avessi previsto tutto questo'
phrase_1 is the input of the BERT model, which returns, for example: phrase_2 ='e se io avessi previsto tutto questo' 
Then we mask the second element of the output of BERT, phrase_2: 'e [MASK] io avessi previsto tutto questo'. 
The process is repeated iteratively for each element of the verse, trasforming it in a different verse.





 
 








