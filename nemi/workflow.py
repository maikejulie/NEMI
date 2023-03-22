import umap
import pickle
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
# import sciris as sc

__all__ = ['NEMI']

default_params = dict(
    embedding_dict = dict(min_dist=0.0, n_components=3, n_neighbors=20),
    clustering_dict = dict(linkage='ward',  n_clusters=30, n_neighbors=40)
)


class SingleNemi():

    def __init__(self, params=None):

        # pipeline parameters
        # self.params = sc.mergedicts(default_params, params)
        # pipeline parameters
        self.params = copy.deepcopy(default_params)
        self.params.update(params if params is not None else {})

        # set during the run
        self.embedding = None
        self.clusters = None
        self.X = None

        return
    
    def run(self, X, save_steps=True):
        '''
        Run the NEMI pipeline

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The data. 
        n : {int}, optional, default=1
            Number of iterations to run
        '''

        # fit the embedding
        print('Fitting the embedding')
        self.fit_embedding(X)

        # predict the clusters
        print('Predicting the clusters')
        self.clusters = self.predict_clusters()

        # sort the clusters by (descending) size
        print('Sorting clusters')
        self.clusters = self.sort_clusters(self.clusters)

    def scale_data(self, X):
        '''
        Scale the data to have a mean and variance of 1.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The data to pick seeds for.

        **kwargs : keyword arguments to embedding function

        '''

        # scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X)
        return scaled_data

    def fit_embedding(self, X):
        '''
        Run the embedding algorithm on the data

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The data to pick seeds for.

        **kwargs : keyword arguments to embedding function

        '''

        # initialize data
        self.X = X
        # run embedding
        self.embedding = self.__embedding_algo(**self.params['embedding_dict'])(self.X)


    def predict_clusters(self):
        '''
        Run the clustering algorithm on the embedding

        Parameters
        ----------
        n_neighbors: {int} default=40
            Number of neighbors for each sample of the kneighbors_graph.
        '''

        return self.__clustering_algo(**self.params['clustering_dict'])(self.X)


    def sort_clusters(self, clusters):
        '''
        Updates cluster labels 0,...,k so that each cluster is of descending size.

        Parameters
        ----------
        clusters : {array, list} 
        '''

        # number of clusters (also the same as the label name in the agglomerated cluster dict)
        n_clusters = np.max(clusters)+1
        #  create a histogram of the different clusters
        hist,_ = np.histogram(clusters, np.arange(n_clusters+1))
        # clusters sorted by size (largest to smallest)
        sorted_clusters= np.argsort(hist)[::-1]
        # assign new labels where labels 0,...,k go in decreasing member size 
        new_labels = np.empty(clusters.shape)
        new_labels.fill(np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label

        return new_labels
        
    def save(self, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)

    def load_embedding(self, filename):
        self.embedding = np.load(filename)

    def save_embedding(self, filename):
        '''

        Parameters
        ----------
        filename : {string} filename to save embedding

        '''
        np.save(filename, self.embedding)

    def plot(self, to_plot=None, **kwargs):
        if to_plot.lower() == 'embedding':
            self._plot_embedding(**kwargs)
        elif to_plot.lower() == 'clusters':
            self._plot_clusters(**kwargs)

    def _plot_embedding(self, s=1, subsample=10, alpha=0.4):

        data = self.embedding

        fig = plt.figure()
        if data.shape[1] == 2:
            ax = plt.gca()
        elif data.shape[1] == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            raise RuntimeError('Embedding not consistent with plotting function')

        ax.scatter(*data[::subsample].T, s=s, alpha=alpha, zorder=4)

    def _plot_clusters(self, n=None, s=1, subsample=10, alpha=0.4):

        self._plot_embedding(s=s, subsample=subsample, alpha=alpha)

        data = self.embedding
        ax = plt.gca()
        labels = self.clusters
        unique_labels = np.sort(np.unique(labels))
        colors = [plt.cm.tab20(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = data[class_member_mask, :]
            ax.scatter(*xy[::subsample].T, c=np.array(col).reshape((1,-1)), s=s, alpha=1, zorder=4)      


    def __embedding_algo(self, **kwargs):
        return umap.UMAP(**kwargs).fit_transform

    def __clustering_algo(self, **kwargs):
        '''
        Parameters
        ----------
        n_neighbors: {int} default=40
            Number of neighbors for each sample of the kneighbors_graph.        
        '''
        # Create a graph capturing local connectivity. Larger number of neighbors
        # will give more homogeneous clusters to the cost of computation
        # time. A very large number of neighbors gives more evenly distributed
        # cluster sizes, but may not impose the local manifold structure of
        # the data
        knn_graph = kneighbors_graph(self.embedding, kwargs['n_neighbors'], include_self=False)
        model = AgglomerativeClustering(linkage=kwargs['linkage'],
                                            connectivity=knn_graph,
                                            n_clusters=kwargs['n_clusters'])
        return model.fit_predict                          


class NEMI(SingleNemi):

    def __init__(self, params=None):
        # pipeline parameters
        self.params = copy.deepcopy(default_params)
        self.params.update(params if params is not None else {})
        self.base_id = None

    def run(self, X, n=1):

        if n == 1:
            super().run(X)
        else:
            # initialize the pack
            nemi_pack = []
            # run the pack
            for member in tqdm(np.arange(n)):
                # create nemi instance
                nemi = SingleNemi(params=self.params)
                # run single instance
                nemi.run(X)        
                # add to the pack
                nemi_pack.append(nemi)

            self.nemi_pack = nemi_pack

        self.assess_overlap()

    def plot(self, to_plot=None, plot_ensemble=False, **kwargs):

        if plot_ensemble:
            for nemi in self.nemi_pack:
                nemi.plot(to_plot, **kwargs)

        if to_plot == 'clusters':
            super().plot('clusters')

    def assess_overlap(self, base_id:int =0, max_clusters=None, **kwargs):
        '''

        Parameters
        ----------
        base_id : {int} optional, default=0
            index (staring at 0) of ensemble member to use as the base comparison

        '''

        self.base_id = base_id
        self.embedding = self.nemi_pack[base_id].embedding

        # list of ensemble members we are comparing to the base
        compare_ids = [i for i in range(len(self.nemi_pack))]
        compare_ids.pop(base_id)

        # identify clusters from the base ensemble member
        base_labels = self.nemi_pack[base_id].clusters

        # number of clusters
        num_clusters = int(np.max(base_labels) + 1)

        # if not pre-set, set max number of clusters to total number of clusters in the base
        if max_clusters is None:
            max_clusters = num_clusters

        sortedOverlap=np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0]))*np.nan

        print(num_clusters, max_clusters)
        summaryStats=np.zeros((num_clusters, max_clusters))

        # compile sorted cluster data
        # TODO: add assert statement to make sure that the clusters have been sorted?
        dataVector=[nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]

        # loop over ensemble members, not including the base member
        for compare_cnt, compare_id in enumerate(compare_ids):
            # grab clusters of ensemble member
            compare_labels= dataVector[compare_cnt]

            # go through each cluster in the base and assess the percentage overlap
            # for every cluster in the ensemble member (overlap / total coverage area) 
            for c1 in range(max_clusters): 
                # Initialize dummy array to mark location of the cluster for the base member
                data1_M = np.zeros(base_labels.shape, dtype=int)
                # mark where the considered cluster is in the member that is being used as the baseline
                data1_M[np.where(c1==base_labels)] = 1 
                # # Count numer of entries [Why?] 
                summaryStats[0, c1]=np.sum(data1_M) 

                # go through each cluster
                # k = 0
                for c2 in range(num_clusters):
                    # Initialize dummy array to mark where the cluster is in the comparison member
                    data2_M = np.zeros(base_labels.shape, dtype=int) 

                    # mark where the considered cluster is in the member that is being used as the comparison
                    data2_M[np.where(c2==compare_labels)] = 1    

                    # Sum of flags where the two datasets of that cluster are both present
                    num_overlap=np.sum(data1_M*data2_M)       

                    #Sum of where they overlap
                    num_total=np.sum(data1_M | data2_M)       

                    #Collect the number that is largest of k and the num_overlap/num_total
                    # k = max(k, num_overlap / num_total)       
                    summaryStats[c2, c1]=(num_overlap / num_total)*100 # Add percentage of coverage

                #Filled in 'summaryStatistics' matrix results of percentage overlaps

            usedClusters = set() # Used to mak sure clusters don't get selected twice
            #Clusters are already sorted by size
            
            sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
            # go through clusters from (biggest to smallest since they are sorted)
            for c1 in range(max_clusters):  
                sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
                #print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

                # find biggest cluster in first column, making sure it has not been used
                sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
                biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

                # record it for later
                usedClusters.add(biggestCluster)

                # Initialize dummy array
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # Select which country is being assessed
                data2_M[np.where(biggestCluster == compare_labels)]=1 # Select cluster being assessed

                sortedOverlapForOneCluster[np.where(data2_M==1)]=1
                sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

        # fill in the base entry in the sorted overlap
        for c1 in range(max_clusters):  
            sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

        # majority vote
        aggOverlaps = np.nansum(sortedOverlap,axis=0)
        voteOverlaps = np.argmax(aggOverlaps,axis=0)

        # save clusters estimated from the ensemble
        self.clusters = voteOverlaps