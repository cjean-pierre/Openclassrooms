import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from wordcloud import WordCloud
import nltk

nltk.download('omw-1.4')

def tokenizer(product_desc):
    """ preprocess product description
    Args :
        product_desc: a string containing a product description
    Output :
        tokens : a list of tokens corresponding to the preprocessed product description

    """
    # Create list of stopwords and punctuation
    stop_words = stopwords.words('english')
    punctuations = string.punctuation + '�' + '…' + '...' + '’' + '±' + '”' + '–' + '“' + '•'+"''"

    # clean text, remove tabs and spaces
    product_desc = product_desc.strip()
    # converting sentence into lowercase
    product_desc = product_desc.lower()
    # replacing specific chars by spaces as word_tokenize ignore them when between two letters
    for char in ['.', ',', '-', '/', '�']:
        product_desc = product_desc.replace(char, ' ')

    # Create list of tokens
    tokens = word_tokenize(product_desc, language='english')

    # Removing stop words and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations]

    # Lemmatizing each token
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # return preprocessed list of tokens
    return tokens


def reduction_SVD(X, n_components, n_iter=7, verbose=True):
    """ perform truncated singular value decomposition and calculate explained variance ratio
    Args :
        X: a matrix of shape (number of product descriptions, dimension of the product description vector)
        n_components : number of components after reduction
    Output :
        X_reduced : the matrix after reduction

    """
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter)
    svd.fit(X)
    evr = round(svd.explained_variance_ratio_.sum() * 100, 2)
    X_reduced = svd.transform(X)

    if verbose:
        print('Pourcentage de variance expliquée pour {} components: {}%'.format(n_components, evr))

        print('Dimension après réduction de dimension:', X_reduced.shape)

    return X_reduced


def reduction_pca(X, n_components, verbose=True):
    """ perform pca  and calculate explained variance ratio
    Args :
        X: a matrix of shape (number of product descriptions, dimension of the product description vector)
        n_components : number of components after reduction
    Output :
        X_reduced : the matrix after reduction

    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    evr = round(pca.explained_variance_ratio_.sum() * 100, 2)
    X_reduced = pca.transform(X)
    if verbose:
        print('Pourcentage de variance expliquée pour {} components: {}%'.format(X_reduced.shape[1], evr))

        print('Dimension après réduction de dimension:', X_reduced.shape)

    return X_reduced


def kmeans_clustering(X, n_clusters):
    """ perform kmean clustering and display number of sample per cluster
    Args:
        X : array on which KMeans is going to be trained
        n_clusters : number of clusters
    output
        X_kmeans : fitted X
        labels : labels of each sample
    """
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    labels = kmeans.labels_
    X_kmeans = kmeans.transform(X)

    # displaying density per cluster
    table = pd.Series(labels).value_counts().to_frame().rename(columns={0: 'Num prod_desc per cluster'})
    table['%'] = round(table['Num prod_desc per cluster'] / table['Num prod_desc per cluster'].sum(axis=0) * 100, 2)
    display(table)

    return X_kmeans, labels


def plot_cats(X, true_labels, cat_names):
    """ plot sample in two dimensions and visualize true categories"""
    fig = plt.figure(figsize=(15, 6))

    # representation des catégories réelles
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='Set1', alpha=0.75)
    ax.legend(handles=scatter.legend_elements()[0], labels=cat_names, loc="best", title="Clusters",
              bbox_to_anchor=(0.5, -0.1), frameon=0)
    plt.title('Représentation des produits par catégories réelles')


def plot_and_compare(X, labels, true_labels, cat_names):
    """ visualize true categories versus labels in two dimension and calculate ARI """
    fig = plt.figure(figsize=(15, 6))

    # representation des catégories réelles
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='Set1', alpha=0.75)
    ax.legend(handles=scatter.legend_elements()[0], labels=cat_names, loc="best", title="Clusters",
              bbox_to_anchor=(0.5, -0.1), frameon=0)
    plt.title('Représentation des produits par catégories réelles')

    # titre global
    ax.text(s='Comparaison catégories réelles/clusters', x=X[:, 0].max() - 20, y=X[:, 1].max() + 20,
            fontsize=14, color="black", fontstyle='normal', fontweight='bold')

    # representation des catégories du clustering
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='Set1', alpha=0.75)
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters",
              bbox_to_anchor=(0.9, -0.1), frameon=0)
    plt.title('Représentation des produits par clusters')

    plt.show()

    # calcul et annotation de l'ARI
    ari = round(adjusted_rand_score(true_labels, labels), 4)

    print('----------------\n')
    print('ARI score: ', ari)


def cloud_img(vocab):
    """ return wordcloud image"""
    wordcloud1 = WordCloud(
        background_color='black',
        width=800,
        height=600, colormap='YlOrRd').generate(" ".join(vocab))
    return wordcloud1
