from visualization_function import create_scatterplots
from sklearn.manifold import TSNE, MDS


class TSNE_abstract():
    model = None
    
    def __init__(self,n_components=2):
        self.model = TSNE(n_components=n_components)
    
    def fit_transform(self,selected_features_scaled):
        return self.model.fit_transform(selected_features_scaled)
    
    def transform(self,selected_features_scaled):
        return self.model.fit_transform(selected_features_scaled)
    
    
class MDS_abstract():
    model = None
    
    def __init__(self, n_components=2):
        self.model = MDS(n_components=n_components)
        
    def fit_transform(self, selected_features_scaled):
        return self.model.fit_transform(selected_features_scaled)
    
    def transform(self, selected_features_scaled):
        return self.model.fit_transform(selected_features_scaled)

model_name = [
    'ica_model.pkl',
    'isomap_model.pkl',
    'kpca_model.pkl',
    'lda_model.pkl',
    'pca_model.pkl',
    'RP_model.pkl',
    'tsne_model.pkl',
    'umap_model.pkl',
    'lle_model.pkl',
    'mds_model.pkl',
    'nmf_model.pkl'
] 


for n in model_name:
    model_dir = fr'azure-ml-internal-project\pipeline\training\models\{n}'
    
    if __name__ == '__main__':
        create_scatterplots(model_dir)
        