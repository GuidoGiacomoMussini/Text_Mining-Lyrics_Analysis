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

### Lyrics preprocessing
First of all, song lyrics contained comments(like the phrase 'you might also like' in the middle of almost each song), header that have to be removed. Here is an example: 
*10 ContributorsCoda di Lupo Lyrics[Testo di "Coda di lupo"] [Strofa 1] Quando ero piccolo m'innamoravo di tutto, correvo dietro ai cani E da marzo a febbraio mio nonno vegliava Sulla corrente di cavalli e di buoi Sui fatti miei e sui fatti tuoi E al dio degli inglesi non credere mai [Strofa 2] [...] You might also like[Strofa 5] Poi tornammo in Brianza per l'apertura della caccia al bisonte [...] e a un dio E a un dio E a un dio senza fiato non credere maiEmbed*

After the processing: 
*Quando ero piccolo m\'innamoravo di tutto, correvo dietro ai cani/  E da marzo a febbraio mio nonno vegliava/  Sulla corrente di cavalli e di buoi/  Sui fatti miei e sui fatti tuoi/  [...]  Poi tornammo in  Brianza per l\'apertura della caccia al bisonte/ [...] E a un dio senza fiato non credere mai*

### Outliers
The next step was to remove songs containing 'outlier verses' which are nothing more than pieces of prose within songs (common in Gaber). This procedure removed around 40 songs.

### Text Preprocessing
In this section have been removed punctuation, capitalization, any special accents or symbols, and even songs written in a foreign language or dialect (common in De André). This step was not entirely successful; all the libraries tried failed to remove the entirety of the foreign language songs. In the end, I opted to use [langdetect](https://pypi.org/project/langdetect/)

After these processes, i obtained: *quando ero piccolo m'innamoravo di tutto correvo dietro ai cani/ e da marzo a febbraio mio nonno vegliava/ sulla corrente di cavalli e di buoi/ sui fatti miei e sui fatti tuoi*, where '/' is the symbol used to divide the verses.

In the end another column was created in the dataset, with the lyrics after the stopwords were removed and the text lemmatized. This is used in the 'Topic Detection' Section.

### Divina Commedia Preprocessing. 
The data is taken by (http://it.wikisource.org). 
This text was used to train a classification algorithm that can recognize whether two words rhyme with each other. The Divina Commedia was chosen because it is a very extensive text, about 14,000 verses, and the rhyme structure is well known.
In this part, the text has been cleaned as the lyrics: comments, punctuation and non-ASCII symbols were removed. 
Then, the dataset used to train the classification algorithm was then built: it contains 2 columns, the first one contains the last word of each verse, the other column contains a numerical indication for the rhyme strucutre: ABA BCB CDC -> 010 121 232.

# Topic Detection







