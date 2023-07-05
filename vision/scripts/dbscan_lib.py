#!/home/nata-brain/miniconda3/envs/eyegaze/bin/python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class dbscanAlgo:
    def __init__(self):
        self.df                 = pd.read_csv('../dataset/gaze_points.csv')
        self.X_train            = self.df[['x', 'y']]
        self.clustering         = DBSCAN(eps=12.5, min_samples=4).fit(self.X_train)
        self.DBSCAN_dataset     = self.X_train.copy()
        self.outliers           = ''
        self.img                = plt.imread("../test_images/grocery-412912_1920.jpg")
        self.path_save          = '../report/'
        self.path_              = '../dataset/'
        self.df_plot            = ''
        
        
    def dfHandling(self):
        self.DBSCAN_dataset.loc[:,'Cluster'] = self.clustering.labels_ 
        self.DBSCAN_dataset.Cluster.value_counts().to_frame()
        self.outliers = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster']==-1]
        self.df = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster'] !=-1]


    def plotClusters(self):    
        plt.figure(figsize=(16, 9))
        p = sns.scatterplot(data = self.df, x = "x", y = "y", hue = self.df.Cluster, legend = "full", palette = "deep")
        sns.move_legend(p, "upper right", bbox_to_anchor = (1.17, 1.), title = 'Clusters')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Clustering'); 
        plt.savefig(f'{self.path_save}clusters.png')        
        self.saveData('clusters', self.df)
        
        
    def saveData(self, label, df):
        if type(label) == type(''):
            # Save df in csv file    
            self.df_plot = pd.DataFrame(df)
            self.df_plot.to_csv(f'{self.path_}/{label}.csv') 
   
        
    def plotGazePoints(self):
        plt.figure(figsize=(16, 9))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.title('Gaze Points')
        plt.savefig(f'{self.path_save}gaze_points.png')
        
        
    def plotDensity(self):
        plt.figure(figsize=(16, 9))
        sns.kdeplot(data = self.df, x="x", y="y", cmap="Reds", fill=True)
        plt.title('Gaze Points Heatmap')
        plt.savefig(f'{self.path_save}heatmap.png')   
        
        
    def overlayImageData(self):
        fig, ax = plt.subplots(figsize = (16, 9))
        p = sns.scatterplot(data = self.df, x = "x", y = "y")
        sns.kdeplot(data = self.df, x = "x", y = "y", cmap = "Reds", fill = True, clip = ((0, 1920), (0, 1080)), alpha=.6)
        ax.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        ax.imshow(self.img, extent=[0, 1920, 0, 1080])
        fig.savefig(f'{self.path_save}overlay_image.png')
        plt.close(fig) 
        
        
    def run(self):
        self.dfHandling()
        self.plotClusters()
        self.plotGazePoints()
        self.plotDensity()
        self.overlayImageData()
    
        
if __name__ == '__main__':
    dbs = dbscanAlgo()
    dbs.run()