# Simple Word Analysis
A program to analyse text on a website and output counts of the words and attempt to cluster the words into n buckets of similar words.
See the [notebook](https://github.com/DrPav/site-word-analysis/blob/master/website-nlp.ipynb) to get an idea of how it works.

![Clusters Graph](https://github.com/DrPav/site-word-analysis/blob/master/graph.PNG)

## Installation
Clone the repository and install the requirements with pip or anaconda (recommended). You need python 3.
The analysis depends on a pre-trained english language model from [Spacy](www.spacy.io) which needs to be downloaded and installed.

### Conda
    conda install --file requirements.txt
    python -m spacy download en_core_web_md

### Pip
    pip install -r requirements.txt
    python -m spacy download en_core_web_md

## Usage
The python file takes 3 positional arguments

1. Number of clusters to put words into
2. path where the output is to be placed
3. url of the website with the text to analyse

The model is very sensitive to the number of clusters, try to pick a number that you think matches the number of word types you expect at the url. A good default is 8. An extra cluster is always crated to hold words that a not recognised in the pre-trained model vocabulary.

### example
    python word-analysis.py 5 example_output https://www.bbc.co.uk/news
    
## Output
A csv file with columns

* word
* word count
* cluster number
* distance to cluster centre
* Decomposition of the word vector into 2-dimensional space using [Principal Component Analysis](https://scikit-learn.org/stable/modules/decomposition.html#pca) (PCA)

A html file containing an interactive chart to explore the data.
    
## Future Work

### Tests and Evaluation
Tests could be written with a framework like pytest to test each component as the project evolves. Since the input to the program is a url and hence continually changing a special test website may have to be set-up that does not change and the word counts are known.

To evaluate the clusters a human would need to evaluate. If the tool is to be used on a certain domain then a training data set could be created giving the expected outputs for given texts. In this case accuracy could be used as a evaluation metric and the problem can become a supervised problem instead of a unsupervised one. The model can also then be improved by fine tuning on a domain specific corpus.

### Database output
SQL Alchemy can be used to output to a relational database. For evaluation cluster fits you would want to store the website text and final clusters every time the program is run. A document store such as mongo would be more appropiate here than SQL.

### Website and API
The core code can be wrapped in a flask API or cloud lambda function. A docker container would be needed to deploy the code with all data science dependencies and the spacy language model. A front end website can also be built once there is an API  that can take a post request with the given parameters and return a json with the data. [Plotly](https://plotly.com/python/) was used to make the plot in python. Plotly also has a [javascript library](https://plotly.com/javascript/).

### Wordcloud
A wordcloud is a popular to visualise word counts. The python [wordcloud](https://github.com/amueller/word_cloud) library is easy to implement.

![Word Cloud Example](https://github.com/amueller/word_cloud/blob/master/examples/constitution.png)
)




