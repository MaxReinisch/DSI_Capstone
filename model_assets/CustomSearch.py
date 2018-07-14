from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances
from sklearn.feature_selection.variance_threshold import VarianceThreshold
from scipy import sparse
import numpy as np
import pandas as pd
import pickle as pkl


dummied_df = pd.read_csv("./model_assets/features_df.csv", index_col='name')
    
with open("./model_assets/content.pkl", "rb") as f:
    content = pkl.load(f)

with open("./model_assets/svd.pkl", "rb") as f:
    svd = pkl.load(f)

with open("./model_assets/vt.pkl", "rb") as f:
    vt = pkl.load(f)




class CustomSearch(object):
    
    features_df = dummied_df
    content = content
    svd = svd
    vt = vt
    
    def __init__(self, games, combination_type = "or"):
        self.boolop = combination_type
        if type(games) == type([]):
            self.search_list=[]
            for game in games:
                title = self.search_game(game)
                self.search_list.append(title)
        else:
            title = self.search_game(games)
            self.search_list = [title]
        
    def search_game(self, search):
        '''
        This helper function looks for games that match the search and returns them as a list
        '''
        return [game for game in CustomSearch.features_df.index if search.lower() in game.lower()][0]
    
    def get_single_feature_vec(self, title):
        '''
        This helper function returns the binary vector associated with the feature space of a single game entry in the dummied dataframe
        '''
        return CustomSearch.features_df.loc[title, :].values
    
    def combine_vec(self, v1, v2, method = 'or'):
        '''
        combines 2 feature vectors in the specified method
        method = {'union', 'and', 'or', 'intersect', 'add'}
        '''
        # add = v1+v2
        # XOR = (v1+v2) %2
        # or = (v1+v2)>0
        # and/intersect = (v1*v2) 

        if method == 'or' or method == 'union':
            return ((v1+v2)>0).astype(int)
        if method == 'and' or method == 'intersect':
            return v1*v2
        if method == 'add':
            return v1+v2

    def getFeatureVector(self):
        ret_vec = self.get_single_feature_vec(self.search_list[0])
        if len(self.search_list) >1:
            for game in self.search_list[1:]:
                ret_vec = self.combine_vec(ret_vec, self.get_single_feature_vec(game))
        return ret_vec
            

    def transform_vector(self, vector):
        '''
        Given a binary vector of features, returns the transformed vector after feature reduction
        '''
        return CustomSearch.svd.transform(CustomSearch.vt.transform(vector.reshape(1, -1)))

    
    # Requires content
    def getCosineToVector(self):
        '''
        returns a vector of cosine distances from a custom transformed vector to every game
        '''
        vector = self.getFeatureVector()
        tvec = self.transform_vector(vector)
        return cosine_distances(tvec, content)

    def SearchSimilarGames(self, n=10):
        '''
        Returns an ordered array of the top n most similar games to the current search feature vector
        '''
        return [game for game in pd.Series(self.getCosineToVector()[0], index=CustomSearch.features_df.index).sort_values().index if game not in self.search_list][0:10]
