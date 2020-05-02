import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
import plotly.express as px
import os, sys

def get_text(url):
    print('Downloading text')
    r = requests.get(url)
    if r.status_code != 200:
        return('Status code: ' + str(r.status_code))
    text = BeautifulSoup(r.content, 'html.parser').get_text()
    return(text)

def nlp_process(text):
    print('Processing text')
    nlp = spacy.load('en_core_web_md', 
        disable=['tagger', 'parser', 'ner']) # We only need tokeniser and the emdeddings
    doc = nlp(text)
    words = [w for w in doc if  w.is_stop != True and w.is_punct != True and w.text not in ['\n', '\n\n' ]]
    words_str = np.array([w.text for w in words])
    words_vec = np.array([w.vector for w in words]) 
    return(words_str, words_vec)

def clustering(X, n):
    print('Clustering words')
    # Handle words not in the vocab that have all zero word vector
    zero_wv = X.sum(axis=1)==0.0
    valid_vectors = X[zero_wv==False]
    kmeans = KMeans(n_clusters=n).fit(valid_vectors)
    centroids = kmeans.cluster_centers_
    clusters = kmeans.predict(valid_vectors)
    distances = []
    for ix, v in enumerate(valid_vectors):
        c = clusters[ix]
        centroid = centroids[c]
        distance = euclidean(v, centroid)
        distances.append(distance)
    # Fill in unkown words as their own cluster -1
    all_clusters = np.full((len(zero_wv)),-1)
    all_clusters[zero_wv==False] = clusters
    all_distances = np.full((len(zero_wv)), 0.0)
    all_distances[zero_wv==False] = distances
    return(all_clusters, all_distances)

def construct_df(url, n_clusters):
    text = get_text(url)
    words_str, words_vec = nlp_process(text)
    clusters, distances = clustering(words_vec, n_clusters)
    pca = PCA(n_components=2).fit_transform(words_vec)
    print('Finalising output')
    df = pd.DataFrame({
        'word': words_str,
        'cluster': clusters,
        'distance_to_cluster_centroid': distances,
        'pca_x': pca[:,0],
        'pca_y': pca[:,1]
        })
    grp = df.groupby(['word','cluster', 'distance_to_cluster_centroid', 'pca_x', 'pca_y'])
    agg = grp.size().reset_index(name='word_count').sort_values('word_count', ascending=False)
    # re-order columns
    agg = agg[['word', 'word_count', 'cluster', 'distance_to_cluster_centroid', 'pca_x', 'pca_y']]
    return(agg)

def cluster_plot(df, path):
    df = df.sort_values('cluster')
    df['cluster'] = df['cluster'].astype(str)
    fig = px.scatter(df, x="pca_x", y="pca_y", color="cluster",
                 hover_data=['word', 'word_count'],
                 color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.write_html(path + 'plot.html')

if __name__ == "__main__":
    n = int(sys.argv[1])
    path = sys.argv[2]
    url = sys.argv[3]

    if path[-1] != '/':
        path = path + '/'
    if os.path.exists(path) != True:
        os.mkdir(path)

    df = construct_df(url, n)
    df.to_csv(path+'data.csv', index=False)
    cluster_plot(df, path)







