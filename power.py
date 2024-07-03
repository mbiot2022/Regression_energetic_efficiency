import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
pd.options.mode.chained_assignment = None # para nao aperecer warnings do pandas

def kmeans(N_g,N_n,N_s=22,L=200,N_r=200):
    
    N_c = N_g.prod()
    gera = default_rng(seed = N_s)
    nos = gera.uniform(0,L, size=(N_n,2,N_r))
    df = [pd.DataFrame({'x':nos[:,0,i],'y':nos[:,1,i]}) for i in range(N_r)]
    
    
    obj = [KMeans(n_clusters=N_c, n_init=20) for i in range(N_r)]
    pred = [obj[i].fit_predict(df[i][['x','y']]) for i in range(N_r)]
    col = len(df[0].columns)
    for i in range(N_r):
        df[i]['label'] = pred[i]
        
    centroid_kmeans = [obj[i].cluster_centers_ for i in range(N_r)]
    for i in range(N_r):
        dif = np.linalg.norm(centroid_kmeans[i][df[i]['label']]- np.array(df[i][['x','y']]), axis = 1)
        df[i]['c_to_node'] = dif
    
    for i in range(N_r):
        df[i]['x_centroid']=centroid_kmeans[i][df[i]['label']][:,0]
        df[i]['y_centroid']=centroid_kmeans[i][df[i]['label']][:,1]   
        
    indice = [[df[j][df[j]['label'] == i]['c_to_node'].idxmin() for i in range(N_c)] for j in range(N_r)]    
     
    df_rc = [[df[j][df[j]['label']==i] for i in range(N_c)] for j in range(N_r)]    
    
    for j in range(N_r):
        for i in range(N_c):
            df_rc[j][i]['x_head'] = df_rc[j][i].loc[indice[j][i]][0]
            df_rc[j][i]['y_head'] = df_rc[j][i].loc[indice[j][i]][1]
            
            
            
    for j in range(N_r):
        for i in range(N_c):
            dif1 = np.array(df_rc[j][i][['x_head','y_head']])- np.array(df_rc[j][i][['x','y']])
            df_rc[j][i]['h_to_node'] = np.linalg.norm(dif1, axis = 1)
    
    df_rc_no_head = [[df_rc[j][i][df_rc[j][i]['x'] != df_rc[j][i].loc[indice[j][i]][0]] for i in range(N_c)] for j in range(N_r)]
    df_rc_heads = [[df_rc[j][i][df_rc[j][i]['x'] == df_rc[j][i].loc[indice[j][i]][0]] for i in range(N_c)] for j in range(N_r)]
     


    for j in range(N_r):
        for i in range(N_c):
            dife_k = np.array(df_rc_heads[j][i][['x_head','y_head']])- np.array([L,L])
            df_rc_heads[j][i]['h_to_sink'] = np.linalg.norm(dife_k, axis = 1)
      


    c = 299792458.0
    freq = 2.4 * 10**9
    P_r = 0.0000001 # convertendo -70 mili decibeis em mili watt
    comprimento = c / freq
    k = P_r *((4 * np.pi)/comprimento)**2 
    
    
    for j in range(N_r):
        for i in range(N_c):
            df_rc_no_head[j][i]['P_t'] = k * (df_rc_no_head[j][i]['h_to_node']**2)
            df_rc_heads[j][i]['P_t'] = k * (df_rc_heads[j][i]['h_to_sink']**2)
    
    spot = [[df_rc_no_head[j][i]['P_t'].sum() for i in range(N_c)]for j in range(N_r)]
    pot = pd.DataFrame(spot)
    
    spot_h_to_s = [[df_rc_heads[j][i]['P_t'].sum() for i in range(N_c)]for j in range(N_r)]
    pot_h_to_s = pd.DataFrame(spot_h_to_s)
    
    return pot,pot_h_to_s    

def grid(N_g,N_n,N_s=22,L=200,N_r=200):
    
    N_c = N_g.prod()
    gera = default_rng(seed = N_s)
    nos = gera.uniform(0,L, size=(N_n,2,N_r))
    df = [pd.DataFrame({'x':nos[:,0,i],'y':nos[:,1,i]}) for i in range(N_r)]
    
    
    encode = OrdinalEncoder()
    gride_code = [np.int32(np.ceil(df[i][["x","y"]]*(N_g/L))).astype(str) for i in range(N_r)]
    for j in range(N_r):
        df[j]['label'] = [''.join(gride_code[j].tolist()[i]) for i in range(N_n)]
    for j in range(N_r):
        df[j]['label'] = encode.fit_transform(np.array(df[j]['label']).reshape(-1,1)).astype(int)
        
   
    index =[np.array([[i,j]for i in range(1,N_g[0]+1)for j in range(1,N_g[1]+1)])for k in range(N_r)]
    centroid = [(index[i] - 0.5) * (L/N_g) for i in range(N_r)]
    # calculando a distancia do centroide ao nó para cada realização
    for i in range(N_r):
        dif = np.linalg.norm(centroid[i][df[i]['label']]- np.array(df[i][['x','y']]), axis = 1)
        df[i]['c_to_node'] = dif
        
    for i in range(N_r):
        df[i]['x_centroid']=centroid[i][df[i]['label']][:,0]
        df[i]['y_centroid']=centroid[i][df[i]['label']][:,1]    

        
    indice = [[df[j][df[j]['label'] == i]['c_to_node'].idxmin() for i in range(N_c)] for j in range(N_r)]   
        
      
    df_rc = [[df[j][df[j]['label']==i] for i in range(N_c)] for j in range(N_r)]
    
    for j in range(N_r):
        for i in range(N_c):
            df_rc[j][i]['x_head'] = df_rc[j][i].loc[indice[j][i]][0]
            df_rc[j][i]['y_head'] = df_rc[j][i].loc[indice[j][i]][1]
            
            
            
    for j in range(N_r):
        for i in range(N_c):
            dif1 = np.array(df_rc[j][i][['x_head','y_head']])- np.array(df_rc[j][i][['x','y']])
            df_rc[j][i]['h_to_node'] = np.linalg.norm(dif1, axis = 1)
    
    df_rc_no_head = [[df_rc[j][i][df_rc[j][i]['x'] != df_rc[j][i].loc[indice[j][i]][0]] for i in range(N_c)] for j in range(N_r)]
    df_rc_heads = [[df_rc[j][i][df_rc[j][i]['x'] == df_rc[j][i].loc[indice[j][i]][0]] for i in range(N_c)] for j in range(N_r)]
    
    for j in range(N_r):
        for i in range(N_c):
            dife_g = np.array(df_rc_heads[j][i][['x_head','y_head']])- np.array([L,L])
            df_rc_heads[j][i]['h_to_sink'] = np.linalg.norm(dife_g, axis = 1)
   
    
    c = 299792458.0
    freq = 2.4 * 10**9
    P_r = 0.0000001 # convertendo -70 mili decibeis em mili watt
    comprimento = c / freq
    k = P_r *((4 * np.pi)/comprimento)**2 
    
    
    for j in range(N_r):
        for i in range(N_c):
            df_rc_no_head[j][i]['P_t'] = k * (df_rc_no_head[j][i]['h_to_node']**2)
            
    for j in range(N_r):
        for i in range(N_c):
            df_rc_heads[j][i]['P_t'] = k * (df_rc_heads[j][i]['h_to_sink']**2)        
    
    spot = [[df_rc_no_head[j][i]['P_t'].sum() for i in range(N_c)]for j in range(N_r)]
    pot = pd.DataFrame(spot)
    
    
    spot_h_to_s = [[df_rc_heads[j][i]['P_t'].sum() for i in range(N_c)]for j in range(N_r)]
    pot_h_to_s = pd.DataFrame(spot_h_to_s)
    return pot,pot_h_to_s