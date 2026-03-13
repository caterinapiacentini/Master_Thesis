#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import faiss
import argparse
import pickle
import os
import random
from numpy.linalg import inv, norm
from wordcloud import WordCloud
from scipy import optimize

# --- LOCK RANDOM SEED FOR REPRODUCIBILITY ---
np.random.seed(42)
random.seed(42)

class SimilarityMeasure():
    def __init__(self, sim_measure='cos_similarity'):
        self.sim_measure = sim_measure
    
    def calc_similarity(self, X, Y):
        if self.sim_measure == 'cos_similarity':
            norm_X = norm(X, axis=1) if X.ndim > 1 else norm(X)
            norm_Y = norm(Y, axis=1) if Y.ndim > 1 else norm(Y)        
            if Y.ndim > X.ndim:
                cs = np.dot(Y,X)/(norm_X*norm_Y)
            else:
                cs = np.dot(X,Y)/(norm_X*norm_Y)       
            return cs
        elif self.sim_measure == 'cos_angle':     
            a = np.sqrt(np.dot(X,X))
            b = np.sqrt(np.dot(Y,Y))
            if a > b:
                cosine = np.arccos(b/a)
            elif b > a:
                cosine = np.arccos(a/b)
            else:
                cosine = 0
            return cosine

class Get_PCA_Embds(object):
    def __init__(self, pca_embds):
        self.pca_embds = pca_embds
    
    def __getitem__(self, keys):
        if isinstance(keys, list):
            return np.vstack([self.pca_embds[key] for key in keys])
        else:
            return self.pca_embds[keys]  

class GTM():
    def __init__(self, model_path, embd_dim=64, nlist=50, nprobe=8):    
        print(f"[INFO] Loading word embeddings from: {model_path}")
        with open(model_path, 'rb') as f:
            pca_embds = pickle.load(f)

        vocab_list           = list(pca_embds.keys())
        self.vocab_series    = pd.Series(vocab_list)
        self.embeddings_dict = Get_PCA_Embds(pca_embds)
        self.cos_angle       = SimilarityMeasure(sim_measure='cos_angle').calc_similarity
        
        self.xb = self.embeddings_dict[vocab_list].astype(np.float32)
        quantizer  = faiss.IndexFlatL2(embd_dim)
        self.index = faiss.IndexIVFFlat(quantizer, embd_dim, nlist)
        self.index.train(self.xb)
        self.index.add(self.xb)  
        self.index.nprobe = nprobe                

    def func(self, a, W_orth, I, X, C, weights, params):
        self.X_new = X + W_orth @ np.diag(a)
        H_A   = self.X_new @ np.linalg.inv(self.X_new.T @ self.X_new) @ self.X_new.T
        RSS   = np.sum((((I-H_A) @ C) @ np.diag(weights))**2)
        return RSS    
            
    def Unitvec(self, v):
        return v/norm(v)
    
    def UnitColumns(self, v):
        return v/norm(v, axis=0)

    def GenWordCloud(self, X):
        topics_dict = {}
        for i, w in enumerate(self.topic):
            v = self.embeddings_dict[w]
            b = np.linalg.inv(X.T@X) @ (X.T @ v)
            v_hat = X @ b
            topics_dict[w] = np.linalg.norm(v_hat)
            
        wordcloud = WordCloud(width=1920, height=1080, max_words=800, relative_scaling=1, normalize_plurals=False, background_color="rgba(255, 255, 255, 1)", mode="RGBA")
        wordcloud = wordcloud.generate_from_frequencies(topics_dict)
        sorted_topics_dict = dict(sorted(topics_dict.items(), key=lambda item: item[1], reverse=True))

        fig = plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"./output/WordClouds/{self.filename}.png", dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight')
        plt.close(fig)  

        return sorted_topics_dict
          
    def run(self, params, pos_seed, neg_seed, topic_name): 
        run, j         = True, 0                   
        proj_subspace  = [pos_seed[i][0] for i,_ in enumerate(pos_seed)]
        
        neg_seed_words = [neg_seed[i][0] for i,_ in enumerate(neg_seed)] if neg_seed else []
        neg_seed_word_str = " ".join(neg_seed_words)
            
        pos_weights = np.array([pos_seed[i][1] for i,_ in enumerate(pos_seed)])        
        neg_weights = np.array([neg_seed[i][1] for i,_ in enumerate(neg_seed)])              
                    
        self.topic = [pos_seed[i][0] for i,_ in enumerate(pos_seed) if pos_seed[i][1] > 0]
        self.filename = f"topic_{topic_name}"
        pos_seed_word_str = " ".join(self.topic)

        os.makedirs('./output/WordClouds', exist_ok=True)

        with open(f"./output/log_{self.filename}.txt", 'w') as f:
            f.write(f"Guided Topic Modeling: {topic_name}\n\
                    \nTopic Size:            {params['cluster_size']}\
                    \nGravity:               {params['gravity']}\
                    \nPositive seed words:   {pos_seed_word_str}\
                    \nNegative seed words:   {neg_seed_word_str}\n\n"
            )        
    
        xq = self.embeddings_dict[proj_subspace+neg_seed_words].astype(np.float32)   
        _, sim_idx = self.index.search(xq, params['k-similar'])        
        bucket_idx = np.unique(sim_idx.flatten())  
       
        V_buckets  = pd.DataFrame(index = self.vocab_series[bucket_idx], data = {'vector': list(self.xb[bucket_idx,:])})
        
        A = None
        for i, w in enumerate(proj_subspace):           
            a = V_buckets.loc[w, 'vector'].reshape(-1,1)
            A = a if i == 0 else np.hstack((A, a))
            V_buckets = V_buckets.drop([w])
        
        N = None
        if len(neg_weights) >= 1:
            for i, w in enumerate(neg_seed_words):
                b = V_buckets.loc[w, 'vector'].reshape(-1,1)
                N = b if i == 0 else np.hstack((N, b))
                V_buckets = V_buckets.drop([w])
     
            if N.ndim == 1 or N.shape[1] == 1:
                N_vector = N.reshape(-1) if N.ndim > 1 else N
                A = self.UnitColumns(A @ np.diag(pos_weights) + np.outer(N_vector, neg_weights))
            else:      
                A = self.UnitColumns(A @ np.diag(pos_weights) + N @ np.diag(neg_weights) @ np.ones(shape=(len(neg_weights), len(proj_subspace)))*(1/len(neg_weights)) )
         
        V = np.vstack(V_buckets.vector).T    
        X, C = A.copy(), A.copy()
        C_orth = np.array([])
        I = np.identity(X.shape[0]) 
        weights = pos_weights   
        gravity = params['gravity']
        
        while run == True:
            j += 1            
            B     = np.linalg.inv(X.T@X) @ X.T @ V
            B_adj = np.diag(pos_weights) @ B                                          
            sel_coeff = B_adj.sum(axis=0) > 0.5*max(pos_weights)           
            V_proj_adj= X @ B_adj[:,sel_coeff] 
            V_orth    = V[:,sel_coeff] - (X @ B[:,sel_coeff])          
            norm_proj = norm(V_proj_adj, axis=0)  
            norm_orth = norm(V_orth, axis=0) 
            alpha     = np.arctan(norm_orth/norm_proj)   
            min_idx   = alpha.argmin()
            true_idx  = np.where(sel_coeff==True)[0] 
            idx       = true_idx[min_idx]                       
            alpha_min = np.min(alpha)                    
            new_word  = V_buckets.index[idx]             
            w = V[:, idx]                                          
            C = np.vstack([C.T, w]).T                    
            self.topic.append(new_word)                  
            
            if ((j-1) % params['update_freq']) == 0:
                w_orth = V_orth[:, min_idx]
            else:
                w_orth = self.Unitvec(w_orth + V_orth[:, min_idx])
                
            weights = weights * (1+gravity)              
            weights = np.append(weights, 1)              
            
            if (j % params['update_freq']) == 0:                               
                W_orth = np.array([w_orth.T]*X.shape[1]).T           
                optimize.minimize(self.func, [0]*X.shape[1], method="CG", args=(W_orth, I, X, C, weights, params)) 
                X = self.UnitColumns(self.X_new)                                                                        
            
            V_buckets = V_buckets.drop([new_word])                       
            V = np.vstack(V_buckets.vector).T    
            gravity = max(0, gravity - params['gravity']/params['cluster_size'])       
                                
            if ((alpha_min > params['alpha_max']) & (j >= 10)) or (C.shape[1] >= params['cluster_size']) or (len(true_idx) == 1):
                run = False

        topics_dict = self.GenWordCloud(X)
        
        df = pd.DataFrame.from_dict(topics_dict, orient='index', columns=["weight"])
        df.to_csv(f'./output/{self.filename}.csv', encoding='utf-8', index=True)

        temp_log = ""
        for i, word in enumerate(df.index):
            temp_log += f"#{i:<3} {word: <30} weight: {df.loc[word, 'weight']:.3f}\n"
        with open(f"./output/log_{self.filename}.txt", 'a') as f:   
            f.write(temp_log) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--combine_grid', action='store_true', help="Combine 6 existing wordclouds into a 2x3 grid")
    parser.add_argument('--topic_name', type=str, required=False, help="Name of the sub-topic")
    parser.add_argument('--model_path', type=str, required=False, help="Path to your Word2Vec .pkl dictionary")
    parser.add_argument('--pos_words', nargs='+', type=str, required=False, help="List of positive seed words")
    parser.add_argument('--pos_weights', nargs='+', type=float, required=False, help="List of weights for positive seeds")
    parser.add_argument('--neg_words', nargs='+', type=str, required=False, default=[], help="List of negative seed words")
    parser.add_argument('--neg_weights', nargs='+', type=float, required=False, default=[], help="List of weights for negative seeds")
    parser.add_argument('--size', type=str, required=False, help="Number of words to collect")
    parser.add_argument('--gravity', type=str, required=False, help="Gravity parameter")
    
    args = parser.parse_args()

    os.makedirs('./output/WordClouds', exist_ok=True)

    # ---------------------------------------------------------
    # GRID COMBINATION MODE
    # ---------------------------------------------------------
    if args.combine_grid:
        print("[INFO] Combining WordClouds into a 2x3 grid...")
        image_files = [
            "topic_Trade_War.png", "topic_Tariffs.png", "topic_Sanctions.png",
            "topic_Embargo.png", "topic_Protectionism.png", "topic_Retaliation.png"
        ]
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()
        
        for idx, img_name in enumerate(image_files):
            img_path = f"./output/WordClouds/{img_name}"
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                axes[idx].imshow(img)
                title = img_name.replace("topic_", "").replace(".png", "").replace("_", " ")
                axes[idx].set_title(title, fontsize=30, pad=20)
            else:
                print(f"[WARNING] Could not find {img_path} for the grid. Subplot left blank.")
            axes[idx].axis("off")
            
        plt.tight_layout()
        plt.savefig("./output/Combined_Geoeconomic_Pressure_Grid.png", dpi=300, facecolor='w', edgecolor='w')
        plt.close(fig)
        print("[OK] Grid successfully saved to ./output/Combined_Geoeconomic_Pressure_Grid.png")
        sys.exit(0)

    # ---------------------------------------------------------
    # NORMAL GTM PROCESSING MODE
    # ---------------------------------------------------------
    if not args.topic_name or not args.model_path or not args.pos_words or not args.pos_weights or not args.size or not args.gravity:
        print("Error: Missing required arguments for GTM run.")
        sys.exit(1)

    print(f'Initialize GTM for topic: {args.topic_name}')
    gtm = GTM(model_path=args.model_path)

    if len(args.pos_words) != len(args.pos_weights):
        print("Error: Positive words must match positive weights.")
        sys.exit(1)
    if len(args.neg_words) != len(args.neg_weights):
        print("Error: Negative words must match negative weights.")
        sys.exit(1)

    pos_seed = list(zip(args.pos_words, args.pos_weights))
    neg_seed = list(zip(args.neg_words, args.neg_weights))

    run_topic = True
    for word_i in pos_seed + neg_seed:
        if word_i[0] not in gtm.vocab_series.values:
            print(f"Error: '{word_i[0]}' is not in the GTM vocabulary! Check for spelling or missing bigram underscores.")
            run_topic = False

    if run_topic:
        params = {
            'cluster_size': float(args.size),      
            'gravity'    :  float(args.gravity),
            'alpha_max':    2.0,      
            'update_freq':  1,        
            'k-similar'  :  5000,     
        }

        print(f'Generate Topic: {args.topic_name}')
        gtm.run(params, pos_seed, neg_seed, args.topic_name)
        print(f"\n[OK] Topic '{args.topic_name}' generation complete!")