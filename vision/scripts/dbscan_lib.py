#!/home/nata-brain/miniconda3/envs/eyegaze/bin/python
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
sns.set(style="whitegrid",rc={"figure.figsize": (19.2, 10.8)})

# Lib to data processing 
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
        
    # Processing dataframe    
    def dfHandling(self):
        self.DBSCAN_dataset.loc[:,'Cluster'] = self.clustering.labels_ 
        self.DBSCAN_dataset.Cluster.value_counts().to_frame()
        self.outliers = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster']==-1]
        self.df = self.DBSCAN_dataset[self.DBSCAN_dataset['Cluster'] !=-1]

    # Creating clusters image to report
    def plotClusters(self):    
        plt.figure(figsize=(19.2, 10.8))
        sns.scatterplot(data = self.df, x = "x", y = "y", hue = self.df.Cluster, legend = "full", palette = "deep")
        plt.savefig(f'{self.path_save}clusters.png')        
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        self.saveData('clusters', self.df)
        
    # Saving new dataframe to future processing    
    def saveData(self, label, df):
        if type(label) == type(''):
            # Save df in csv file    
            self.df_plot = pd.DataFrame(df)
            self.df_plot.to_csv(f'{self.path_}/{label}.csv') 
   
    # Creating gaze points image to report    
    def plotGazePoints(self):
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        plt.savefig(f'{self.path_save}gaze_points.png')
        
    # Creating heatmap overlay image to report    
    def plotDensity(self):
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], 'r', linestyle = '-')
        
        sns.kdeplot(data = self.df, x = "x", y = "y", cmap = "Reds", fill = True, alpha = .6)
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        plt.savefig(f'{self.path_save}heatmap.jpg', dpi = 100)
        
        self.overlayImageData()
        
    # Processing data to overlay image     
    def overlayImageData(self):
        img = np.asarray(Image.open('/home/nata-brain/camera_ws/src/EyeHeadTrack/vision/test_images/grocery.jpg'))
        fig, ax = plt.subplots()
        ax.imshow(img, extent = [0, 1920, 0, 1080])
        sns.kdeplot(data = self.df, x = "x", y = "y", cmap = "Reds", common_norm=False, levels=50, fill = True, alpha = .5)
        plt.xlim(0, 1920)
        plt.ylim(0, 1080)
        fig.savefig(f'{self.path_save}overlay.jpg', dpi = 100)
        
    # Showing image to test       
    def showImage(self):
        cv2.imshow('Image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def run(self):
        self.dfHandling()
        self.plotClusters()
        self.plotGazePoints()
        self.plotDensity()
        
    
        
if __name__ == '__main__':
    dbs = dbscanAlgo()
    dbs.showImage()
    input('Aperte enter para o processamento dos dados: ')
    dbs.run()
    print('------\nDados Processados\n------')