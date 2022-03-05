import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
Customer_data = pd.read_csv("Customer data.csv")
[x,y]=np.shape(Customer_data)
Customer_data=pd.DataFrame(Customer_data.iloc[:,1:y]).to_numpy()
K=5


Centroids_temp=Customer_data[np.random.choice(x,K,replace=False)]
def GUC_Distance ( Cluster_Centroids, Data_points, Distance_Type ):
   
    average_centroids=np.zeros([K,1])
    average_CustData=np.zeros([x,1])
    Dis_Ec=np.zeros([x,])
    Dis_Ec=np.zeros([x,K])
    New_CustData=np.zeros([x,1])
    Newcentroid=np.zeros([K,1])
    subt_centroids=np.zeros([K,y-1])
    subt_CustData=np.zeros([x,y-1])
    numerator=np.zeros([x,K])
    denomenator=np.zeros([x,K])
    if Distance_Type== 'Eclidean':
        for i in range(K):    
            Dis_Ec[:,i] = ((Customer_data-Cluster_Centroids[i])**2).sum(axis=1)
    else:
       
        average_centroids=np.average(Centroids_temp,axis=1)
        avearge_CustData=np.average(Data_points,axis=1)
       
        for k in range(x):
            subt_CustData[k]=Data_points[k]-avearge_CustData[k]
       
        
        for k in range(K):  
            subt_centroids[k]=Centroids_temp[k]-average_centroids[k]
       
        
        for row in range (x):
            New_CustData[row]=((Data_points[row]-average_CustData[row])**2).sum()
        for row1 in range (K):
            Newcentroid[row1]=((Centroids_temp[row1]-average_centroids[row1])**2).sum()
        for u in range(K):
            for j in range(x):
                numerator[j,u]=(subt_CustData[j]*subt_centroids[u]).sum()
                denomenator[j,u]=np.sqrt(New_CustData[j]*Newcentroid[u])
        Dis_Ec=1-(numerator/denomenator)
        print(Dis_Ec)
        print(np.shape(Dis_Ec))
       

    return Dis_Ec
Centroid_t_rand=Customer_data[np.random.choice(x,K,replace=False)]
def GUC_Kmean ( Customer_data_NO_ID, Number_of_Clusters, Distance_Type ):
    k=Number_of_Clusters
    #j function for distortion
    J_min=0
    centroids_f=np.zeros([x,k])
    J=0 #distortion
    J_before=-1
    users_per_cluster=np.zeros([k])
    sumofUser_perCluster=np.zeros([k,y-1])
    for u in range (100):
        Centroid_t_rand=Customer_data_NO_ID[np.random.choice(x,K,replace=False)]
        while True:
            Distance_diff=GUC_Distance(Centroid_t_rand,Customer_data_NO_ID, 'Eclidean' )
            MinimumDistance=np.argmin(Distance_diff,axis=1)
            mindist=np.min(Distance_diff,axis=1)
            for cluster_number in range(k):
                for j in range(x):
                    if MinimumDistance[j]==cluster_number:
                        users_per_cluster[cluster_number]=users_per_cluster[cluster_number]+1
                        sumofUser_perCluster[cluster_number]=sumofUser_perCluster[cluster_number]+Customer_data_NO_ID[j]
                        J=J+((((Customer_data_NO_ID[j]-Centroid_t_rand[cluster_number])**2)).sum())
       
                sumofUser_perCluster[cluster_number]=sumofUser_perCluster[cluster_number]/users_per_cluster[cluster_number]
                J=J/x
            if J==J_before:
                break
            else:
                J_before=J
            if u==0:
                J_min=J
            if J<J_min:
                J_min=J
                Centroid_t_rand=sumofUser_perCluster
           
            return [mindist, J_min,MinimumDistance]
           




(D,c,m)=GUC_Kmean(Customer_data, K, 'Ecledian')
df = pd.DataFrame(dict(x=Customer_data[:,2], y=Customer_data[:,4], label=m.astype(int)))
colors = {0:'blue', 1:'orange', 2:'green',3:'red',4:'olive',5:'yellow',6:'pink'}
fig, ax = plt.subplots(figsize=(8, 8))
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
ax.scatter(Centroid_t_rand[:, 2],Centroid_t_rand[:, 4], marker='*', s=150, c='#ff2222')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()


wcss=[]

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(Customer_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elobw Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()