Deep Learning & Natural Language Processing: Introducing word2vec
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 Python and Pip](#01-python--pip)
    + [0.2 Libraries](#02-libraries)
- [1.0 Background](#10-background)
	+ [1.1 Word Embeddings](#10-word-embeddings)
	+ [1.2 Word2vec](#12-word2vec)
	+ [1.3 Skip-gram Model](#13-skip--gram-model)
		* [1.3.1 Inputs](#131-inputs)
		* [1.3.2 Output](#132-output)
	+ [1.4 Autoencoders](#14-data-types)
	+ [1.5 Fine-Tuning](#15-fine--tuning)
	+ [1.6 Glove](#16-glove)
	+ [1.7 Vector Search Spaces](#17-vector-search-spaces)
- [2.0 LSTM Networks](#20-lstm-networks)
- [3.0 gensim & nltk](#30-gensim--nltk)
- [4.0 Convolution Neural Networks](#40-convolution-neural-networks)
	+ [4.1 Input](#41-input)
	+ [4.2 Filters](#42-filters)
- [5.0 Final Words](#60-final-words)
	+ [5.1 Resources](#61-resources)


## 0.0 Setup

This guide was written in Python 3.5.

### 0.1 Python and Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 Libraries

```
pip3 install tensorflow
pip3 install gensim
pip3 install word2vec
```


## 1.0 Background

### 1.1 Word Embeddings

Image and audio processing systems work with high-dimensional datasets encoded as vectors of the raw pixel-intensities for image data. For tasks like speech recognition we know that all the information required to successfully perform the task is encoded in the data. However, natural language processing systems traditionally treat words as discrete atomic symbols, and therefore 'cat' may be represented as Id537 and 'dog' as Id143. These encodings are arbitrary, and provide no useful information to the system regarding the relationships that may exist between the individual symbols. 

This means that the model can leverage very little of what it has learned about 'cats' when it is processing data about 'dogs' (such that they are both animals, four-legged, pets, etc.). Representing words as unique, discrete IDs furthermore leads to data sparsity, and usually means that we may need more data in order to successfully train statistical models. However, using vector representations can overcome some of these obstacles.

 

### 1.2 Word2Vec

Word2vec is a computationally efficient predictive model for learning word embeddings from raw text. It comes in two models: the Continuous Bag-of-Words model (CBOW) and the Skip-Gram model. Algorithmically, these models are similar, except that CBOW predicts target words (e.g. 'mat') from source context words ('the cat sits on the'), while the skip-gram does the inverse and predicts source context-words from the target words. 

The easiest way to think about word2vec is that it figures out how to place words on a graph in such a way that their location is determined by their meaning. In other words, words with similar meanings will be clustered together. More interestingly, though, is that the gaps and distances on the graph have meaning as well. If you go to where “king” appears, and move the same distance and direction between “man” and “woman”, you end up where “queen” appears. And this is true of all kinds of semantic relationships! You can look at this visualization [here](https://www.tensorflow.org/versions/master/images/linear-relationships.png).

Put more mathematically, you can think of this relationship as: 

```
[king] - [man] + [woman] ~= [queen]
```


### 1.3 Skip-gram Model


#### 1.3.1 Inputs

The input of the skip-gram model is a single word, `w_n`. For example, in the following sentence:

``` 
I drove my car to school.
```

"car" could be the input, or "school". 


#### 1.3.2 Outputs

The output of a skip-gram model is the words in `w_n`s context. Going along with the example from before, the output would be:

``` 
{"I","drove","my","to","school"}
```

This output is defined as `{w_O,1 , ... , w_O,C }`, where C is the word window size that you define. 


### 1.4 Autoencoders


Autoencoders are a kind of neural network designed for dimensionality reduction; in other words, representing the same information with fewer numbers. A wide range of autoencoder architectures exist, including Denoising Autoencoders, Variational Autoencoders, or Sequence Autoencoders. 

The basic premise is simple — we take a neural network and train it to spit out the same information it’s given. By doing so, we ensure that the activations of each layer must, by definition, be able to represent the entirety of the input data. 

#### 1.4.1 How do Autencoders work? 

If each layer is the same size as the input, this becomes trivial, and the data can simply be copied over from layer to layer to the output. But if we start changing the size of the layers, the network inherently learns a new way to represent the data. If the size of one of the hidden layers is smaller than the input data, it has no choice but to find some way to compress the data.

And that’s exactly what an autoencoder does. The network starts out by “compressing” the data into a lower-dimensional representation, and then converts it back to a reconstruction of the original input. If the network converges properly, it will be a more compressed version of the data that encodes the same information. 

It’s often helpful to think about the network as an “encoder” and a “decoder”. The first half of the network, the encoder, takes the input and transforms it into the lower-dimensional representation. The decoder then takes that lower-dimensional representation and converts it back to the original image (or as close to it as it can get). The encoder and decoder are still trained together, but once we have the weights, we can use the two separately — maybe we use the encoder to generate a more meaningful representation of some data we’re feeding into another neural network, or the decoder to let the network generate new data we haven’t shown it before.

#### 1.4.2 Why are Autoencoders Important?

Because some of your features may be redundant or correlated, this can result in wasted processing time and overfitting in your model! Autoencoders help us avoid that pitfall!



### 1.5 Fine-Tuning

Fine-Tuning refers to the technique of initializing a network with parameters from another task (such as an unsupervised training task), and then updating these parameters based on the task at hand. For example, NLP architecture often use pre-trained word embeddings like word2vec, and these word embeddings are then updated during training based for a specific task like Sentiment Analysis.

### 1.6 Glove

GloVe is an unsupervised learning algorithm for obtaining vector representations (embeddings) for words. GloVe vectors serve the same purpose as word2vec but have different vector representations due to being trained on co-occurrence statistics.


### 1.7 Vector Space Models

A vector space search involves converting documents into vectors, where each dimension within the vectors represents a term. VSMs represent embedded words in a continuous vector space where semantically similar words are mapped to nearby points. 

#### 1.7.1 Distributional Hypothesis

VSMs methods depend highly on the Distributional Hypothesis, which states that words that appear in the same contexts share semantic meaning. The different approaches that leverage this principle can be divided into two categories: count-based methods and predictive methods. 


#### 1.7.2 Stemming & Stop Words

To begin implementing a Vector Space Model, first you must get all terms within documents and clean them. A tool you will likely utilize for this process is a stemmer, which you might recall takes words and reduces them to the base (or unchanging portion). Words which have a common stem often have similar meanings, which is why you'd likely want to utilize a stemmer.

We will also want to remove any stopwords, such as a, am, an, also ,any, and, etc. Stop words have little value in this search so it's better that we just get rid of them. 

Just as we've seen before, this is what that code might look like:

``` python
 self.stemmer = PorterStemmer()

def removeStopWords(self,list):
	return([word for word in list if word not in self.stopwords])


def tokenise(self, string):
	string = self.clean(string)
    words = string.split(" ")
	return ([self.stemmer.stem(word,0,len(word)-1) for word in words])
```

#### 1.7.3 Map Keywords to Vector Dimensions

Next, we'll map the vector dimensions to keywords using a dictionary.

``` python
def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        # Mapped documents into a single word string
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        # Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
                vectorIndex[word]=offset
                offset+=1
        return vectorIndex  #(keyword:position)
```

#### 1.7.4 Map Document Strings to Vectors

We use the Simple Term Count Model to map the documents 

``` python
def makeVector(self, wordString):

        # Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
                vector[self.vectorKeywordIndex[word]] += 1; # Use Simple Term Count Model
        return vector
```

#### 1.7.5 Find Related Documents

We now have the ability to find related documents. We can test if two documents are in the same concept space by looking at the the cosine of the angle between the document vectors. We use the cosine of the angle as a metric for comparison. If the cosine is 1 then the angle is 0° and hence the vectors are parallel (and the document terms are related). If the cosine is 0 then the angle is 90° and the vectors are perpendicular (and the document terms are not related).

We calculate the cosine between the document vectors in python using scipy.

``` python
def cosine(vector1, vector2):
        """ related documents j and q are in the concept space by comparing the vectors :
                cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
        return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))
```

#### 1.7.6 Search Vector Space

In order to perform a search across keywords we need to map the keywords to the vector space. We create a temporary document which represents the search terms and then we compare it against the document vectors using the same cosine measurement mentioned for relatedness.

``` python
def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        ratings.sort(reverse=True)
        return ratings
```


## 2.0 LSTM Networks

Long Short Term Memory Networks are a special kind of recurrent neural network capable of learning long-term dependencies. LSTMs are explicitly designed to avoid the long-term dependency problem, so remembering information for long periods of time is usually their default behavior.

LSTMs also have the same RNN chain-like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very particular way. You can see that [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png). Compared to the typical recurrent neural network [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png), you can see there's a much more complex process happening. 


### 2.1 First Step

The first step in a LSTM is to decide what information is going to be thrown away from the cell state. This decision is made by a <b>sigmoid layer</b> called the “forget gate layer.” It outputs a number between 00 and 11 for each number in the cell state, where A 1 represents disposal and a 0 means storage.

In the context of natural language processing, the cell state would include the gender of the present subject, so that the correct pronouns can be used in the future.

### 2.2 Second Step

This next step is deciding what new information is going to be stored in the cell state. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Secondly, a tanh layer creates a vector of new values that <i>could</i> be added to the cell state. 

In our NLP example, we would want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.

### 2.3 Third Step

So now we update the old cell state. The previous steps already decided what to do, but we actually do it in this step.

We multiply the old state by new state function, causing it to forget what we've learned earlier. Then we add the updates!

In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, which we decided in the previous steps.

### 2.4 Final Step

Finally, we need to decide what we’re going to output. This output will be based on our cell state, but only once we've filtered it. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanhtanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we need to.

For our NLP example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.

## 3.0 gensim & nltk

`gensim` provides a Python implementation of Word2Vec that works great in conjunction with NLTK corpora. The model takes a list of sentences as input, where each sentence is expected to be a list of words. 

Let's begin exploring the capabilities of these two modules and import them as needed:

``` python
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
```

Next, we'll load the Word2Vec model on three different NLTK corpora for the sake of comparison. 

``` python
b = Word2Vec(brown.sents())
mr = Word2Vec(movie_reviews.sents())
t = Word2Vec(treebank.sents())
```

Here we'll use the word2vec method `most_similar` to find words associated with a given word. We'll call this method on all three corpora and see how they differ. Note that we’re comparing a wide selection of text from the brown corpus with movie reviews and financial news from the treebank corpus.

``` python
print(b.most_similar('money', topn=5))
print(mr.most_similar('money', topn=5))
print(t.most_similar('money', topn=5))
```

And we get:

``` 
[(u'care', 0.9144914150238037), (u'chance', 0.9035866856575012), (u'job', 0.8981726765632629), (u'trouble', 0.8752247095108032), (u'everything', 0.8740494251251221)]

[(u'him', 0.783672571182251), (u'chance', 0.7683249711990356), (u'someone', 0.76824951171875), (u'sleep', 0.7554738521575928), (u'attention', 0.737582802772522)]

[(u'federal', 0.9998856782913208), (u'all', 0.9998772144317627), (u'new', 0.9998764991760254), (u'only', 0.9998745918273926), (u'companies', 0.9998691082000732)]
```

The results are vastly different! Let's try a different example: 
``` python
print(b.most_similar('great', topn=5))
print(mr.most_similar('great', topn=5))
print(t.most_similar('great', topn=5))
```

Again, still pretty different!
```
[(u'common', 0.8809183835983276), (u'experience', 0.8797307014465332), (u'limited', 0.841762900352478), (u'part', 0.8300108909606934), (u'history', 0.8120908737182617)]

[(u'nice', 0.8149471282958984), (u'decent', 0.8089801073074341), (u'wonderful', 0.8058308362960815), (u'good', 0.7910960912704468), (u'fine', 0.7873961925506592)]

[(u'out', 0.9992268681526184), (u'what', 0.9992024302482605), (u'if', 0.9991806745529175), (u'we', 0.9991364479064941), (u'not', 0.999133825302124)]
```

And lastly,

``` python
print(b.most_similar('company', topn=5))
print(mr.most_similar('company', topn=5))
print(t.most_similar('company', topn=5))
```

It’s pretty clear from these examples that the semantic similarity of words can vary greatly depending on the textual context. 


## 4.0 Convolution Neural Networks

Convolutional Neural Network (CNNs) are typically thought of in terms of Computer Vision, but they have also had some important breakthroughs in the field of Natural Language Processing. In this section, we'll overview how CNNs can be applied to Natural Language Processing instead.

### 4.1 Input

Instead of image pixels, the input for an NLP model will be sentences or documents represented as a matrix. Each row of the matrix corresponds to one token. These vectors will usually be word embeddings, discussed in section 1 of this workshop, like word2vec or GloVe, but they could also be one-hot vectors that index the word into a vocabulary. For a 10 word sentence using a 100-dimensional embedding we would have a 10×100 matrix as our input. That's what replaces our typical image input.

### 4.2 Filters

Previously, our filters would slide over local patches of an image, but in NLP we have these filters slide over full rows of the matrix (remember that each row is a word/token). This means that the “width” of our filters will usually be the same as the width of the input matrix. The height may vary, but sliding windows over 2-5 words at a time is typical. 

Putting all of this together, here's what an NLP Convolution Neural Network would look [like](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-12.05.40-PM.png). 


## 5.0 Final Words

If you want to learn more about Deep Learning & Natural Language Processing, check out our meetups for future content! You can also check out the resources section of this workshop for more! 


### 5.1 Resources

[Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398) <br>
[Glove](http://nlp.stanford.edu/projects/glove/)


### 5.2 More!

Join us for more workshops! 

[Wednesday, December 21st, 6:00pm: Intro to Data Science](https://www.meetup.com/Byte-Academy-Finance-and-Technology-community/events/236199419/) <br>
[Thursday, December 22nd, 6:00pm: Intro to Data Science with R](https://www.meetup.com/Byte-Academy-Finance-and-Technology-community/events/236199452/) <br>
[Tuesday, December 27th, 6:00pm: Python vs R for Data Science](https://www.meetup.com/Byte-Academy-Finance-and-Technology-community/events/236203310/)

