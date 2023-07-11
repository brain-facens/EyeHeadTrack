#!/home/nata-brain/miniconda3/envs/eyegaze/bin/python
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class dbscanAlgo:
    def __init__(self):
        self.df                 = pd.read_csv('/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/dataset/gaze_points.csv') 
        self.X_train            = self.df[['x', 'y']]
        self.clustering         = DBSCAN(eps=12.5, min_samples=4).fit(self.X_train)
        self.DBSCAN_dataset     = self.X_train.copy()
        self.outliers           = ''
        self.img                = plt.imread("/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/test_images/grocery.jpg")
        self.path_save          = '/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/report/'
        self.path_              = '/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/dataset/'
        self.df_plot            = ''
        
        
    def dfHandling(self):
        self.DBSCAN_dataset.loc[:,'Cluster'] = self.clustering.labels_ 
        self.DBSCAN_dataset.Cluster.value_counts().to_frame()
        self.outliers = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster']==-1]
        self.df = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster'] !=-1]


    def plotClusters(self):    
        plt.figure(figsize=(19.2, 10.8))
        sns.scatterplot(data = self.df, x = "x", y = "y", hue = self.df.Cluster, legend = "full", palette = "deep")
        plt.savefig(f'{self.path_save}clusters.png')        
        self.saveData('clusters', self.df)
        
        
    def saveData(self, label, df):
        if type(label) == type(''):
            # Save df in csv file    
            self.df_plot = pd.DataFrame(df)
            self.df_plot.to_csv(f'{self.path_}/{label}.csv') 
   
        
    def plotGazePoints(self):
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        plt.savefig(f'{self.path_save}gaze_points.png')
        
        
    def plotDensity(self):
        plt.figure(figsize=(19.2, 10.8))
        sns.scatterplot(data = self.df, x = "x", y = "y")
        sns.kdeplot(data = self.df, x = "x", y = "y", cmap = "Reds", fill = True, alpha = .6)
        #img = cv2.cvtColor(cv2.imread('/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/test_images/grocery.jpg'), cv2.COLOR_BGR2RGB)
        #img = cv2.flip(img, 1)
        #plt.imshow(img)
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        plt.savefig(f'{self.path_save}heatmap.png', dpi = 100)
        #plt.show() 
        
    def overlayImageData(self):
        """ fig = plt.figure(frameon = False)
        sns.scatterplot(data = self.df, x = "x", y = "y")
        sns.kdeplot(data = self.df, x = "x", y = "y", cmap = "Reds", fill = True)
        fig.set_size_inches(1920/100, 1080/100)
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        fig.savefig(f'{self.path_save}overlay_image.png') """
        pass
        
        
    def showImage(self):
        cv2.imshow('Image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def run(self):
        self.dfHandling()
        self.plotClusters()
        self.plotGazePoints()
        self.plotDensity()
        self.overlayImageData()
    
        
if __name__ == '__main__':
    dbs = dbscanAlgo()
    dbs.showImage()
    input('Aperte enter para o processamento dos dados: ')
    dbs.run()
    print('------\nDados Processados\n------')