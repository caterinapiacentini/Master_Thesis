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
import json
import os
from numpy.linalg import inv, norm
from wordcloud import WordCloud
from scipy import optimize
from datetime import datetime

class SimilarityMeasure():
    def __init__(self, sim_measure='cos_similarity'):
        self.sim_measure = sim_measure
    
    def calc_similarity(self, X, Y):
        if self.sim_measure == 'cos_similarity':
            if X.ndim > 1: 
                norm_X = norm(X, axis=1) 
            else: 
                norm_X = norm(X)
            if Y.ndim > 1: 
                norm_Y = norm(Y, axis=1) 
            else: 
                norm_Y = norm(Y)        
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
            
        else:
            print(f"{self.sim_measure} not available!")

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
        self.cos_similarity  = SimilarityMeasure(sim_measure='cos_similarity').calc_similarity          
        
        # Efficient similarity search using Faiss
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
            
        # UPGRADED RESOLUTION: 1920x1080 canvas
        wordcloud = WordCloud(width=1920, height=1080, max_words=800, relative_scaling=1, normalize_plurals=False, background_color="rgba(255, 255, 255, 1)", mode="RGBA")
        wordcloud = wordcloud.generate_from_frequencies(topics_dict)
        sorted_topics_dict = dict(sorted(topics_dict.items(), key=lambda item: item[1], reverse=True))

        # UPGRADED FIGURE SIZE: 16:9 ratio
        fig = plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        
        # UPGRADED DPI: 300 (Print quality)
        plt.savefig(f"./output/WordClouds/{self.filename}.png", dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight')
        plt.close(fig)  

        return sorted_topics_dict
          
    def run(self, params, pos_seed, neg_seed, topic_name=None): 
        run, j         = True, 0                   
        proj_subspace  = [pos_seed[i][0] for i,_ in enumerate(pos_seed)]
        
        neg_seed_word_str = ""
        try:
            neg_seed_words = [neg_seed[i][0] for i,_ in enumerate(neg_seed)]
            for ns in neg_seed_words:
                neg_seed_word_str += ns+"  "
        except:
            neg_seed_words = []
            
        pos_weights = np.array([pos_seed[i][1] for i,_ in enumerate(pos_seed)])        
        neg_weights = np.array([neg_seed[i][1] for i,_ in enumerate(neg_seed)])              
                    
        self.topic = [pos_seed[i][0] for i,_ in enumerate(pos_seed) if pos_seed[i][1] > 0]
         
        # Dynamically create filename based on topic_name from Slurm, or fallback to original
        if topic_name:
            self.filename = f"topic_{topic_name}"
        else:
            topic_name_base = "_".join(self.topic[:min(2, len(self.topic))])
            self.filename = f"topic_{topic_name_base}_" + datetime.now().strftime('%Hh_%Mm_%Ss')

        # Ensure output directories exist
        os.makedirs('./output/WordClouds', exist_ok=True)

        pos_seed_word_str = " ".join(self.topic)

        # Create log file strictly in the output folder
        with open(f"./output/log_{self.filename}.txt", 'w') as f:
            f.write(f"Guided Topic Modeling\n\
                    \nTopic Size:            {params['cluster_size']}\
                    \nGravity:               {params['gravity']}\
                    \nPositive seed words:   {pos_seed_word_str}\
                    \nNegative seed words:   {neg_seed_word_str}\n\n"
            )        
    
        # Similarity Search
        xq = self.embeddings_dict[proj_subspace+neg_seed_words].astype(np.float32)   # query vectors
        _, sim_idx = self.index.search(xq, params['k-similar'])        
        bucket_idx = np.unique(sim_idx.flatten())  
       
        V_buckets  = pd.DataFrame(index = self.vocab_series[bucket_idx], 
                                  data  = {'vector': list(self.xb[bucket_idx,:])})
        
        self.V_bucket = V_buckets
        
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
     
            # Adjust A by the negative seed words
            if N.ndim == 1 or N.shape[1] == 1:
                N_vector = N.reshape(-1) if N.ndim > 1 else N
                A = self.UnitColumns(A @ np.diag(pos_weights) + np.outer(N_vector, neg_weights))
            else:      
                A = self.UnitColumns(A @ np.diag(pos_weights) + N @ np.diag(neg_weights) @ np.ones(shape=(len(neg_weights), len(proj_subspace)))*(1/len(neg_weights)) )
         
        V = np.vstack(V_buckets.vector).T    
        X, C = A.copy(), A.copy()
        C_orth, self.var, self.resid_mean = np.array([]), np.array([]), np.array([])
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
                result = optimize.minimize(self.func, [0]*X.shape[1], method="CG", args=(W_orth, I, X, C, weights, params)) 
                X      = self.UnitColumns(self.X_new)                                                                        
            
            V_buckets = V_buckets.drop([new_word])                       
            V = np.vstack(V_buckets.vector).T    
            
            gravity = max(0, gravity - params['gravity']/params['cluster_size'])       
                        
            if j == 1:
                C_orth = V_orth[:, min_idx]   
                self.resid_sum = np.array([norm(C_orth)])
            else:
                C_orth   = np.vstack([C_orth.T, V_orth[:, min_idx]]).T
                self.var = np.append(self.var, np.var(C_orth, axis=1).mean())
                self.resid_mean = np.append(self.resid_mean, norm(C_orth, axis=0).mean())
                                
            print(f"{new_word: <30} word #{j:<3}; angle: {alpha_min:.3f}")
            if ((alpha_min > params['alpha_max']) & (j >= 10)) or (C.shape[1] >= params['cluster_size']) or (len(true_idx) == 1):
                run = False

        topics_dict = self.GenWordCloud(X)
        topics_dict = pd.DataFrame.from_dict(topics_dict, orient='index', columns=["weight"])
        topics_dict.to_csv(f'./output/{self.filename}.csv', encoding='utf-8', index=True)

        temp_log = ""
        for i, word in enumerate(topics_dict.index):
            temp_log += f"#{i:<3} {word: <30} weight: {topics_dict.loc[word, 'weight']:.3f}\n"

        # Append to the log file strictly in the output folder
        with open(f"./output/log_{self.filename}.txt", 'a') as f:   
            f.write(temp_log) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- ADDED ARGUMENTS FOR MULTI-TOPIC & GRID ---
    parser.add_argument('--combine_grid', action='store_true', help="Combine 6 existing wordclouds into a 2x3 grid")
    parser.add_argument('--topic_name', type=str, required=False, help="Name of the sub-topic")
    
    # --- DYNAMIC LIST ARGUMENTS (Set to required=False so --combine_grid runs cleanly) ---
    parser.add_argument('--model_path', type=str, required=False, help="Path to your Word2Vec .pkl dictionary")
    parser.add_argument('--pos_words', nargs='+', type=str, required=False, help="List of positive seed words (e.g. sanctions tariffs)")
    parser.add_argument('--pos_weights', nargs='+', type=float, required=False, help="List of weights for positive seeds (e.g. 1.0 1.0)")
    parser.add_argument('--neg_words', nargs='+', type=str, required=False, default=[], help="List of negative seed words")
    parser.add_argument('--neg_weights', nargs='+', type=float, required=False, default=[], help="List of weights for negative seeds")
    parser.add_argument('--size', type=str, required=False, help="Number of words to collect")
    parser.add_argument('--gravity', type=str, required=False, help="Gravity parameter (e.g. 1.5)")
    
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
    if not args.model_path or not args.pos_words or not args.pos_weights or not args.size or not args.gravity:
        print("Error: Missing required arguments for GTM run.")
        sys.exit(1)

    run = True
    print('Initialize GTM')
    gtm = GTM(model_path=args.model_path)

    if len(args.pos_words) != len(args.pos_weights):
        print("Error: The number of positive words must match the number of positive weights.")
        sys.exit(1)
    if len(args.neg_words) != len(args.neg_weights):
        print("Error: The number of negative words must match the number of negative weights.")
        sys.exit(1)

    pos_seed = list(zip(args.pos_words, args.pos_weights))
    neg_seed = list(zip(args.neg_words, args.neg_weights))

    for word_i in pos_seed + neg_seed:
        if word_i[0] not in gtm.vocab_series.values:
            print(f"Error: '{word_i[0]}' is not in the GTM vocabulary! Check for spelling or missing bigram underscores.")
            run = False

    if run:
        params = {
            'cluster_size': float(args.size),      
            'gravity'    :  float(args.gravity),
            'alpha_max':    2.0,      
            'update_freq':  1,        
            'k-similar'  :  5000,     
        }

        print('Generate Topic')
        gtm.run(params, pos_seed, neg_seed, topic_name=args.topic_name)
        print("\n[OK] Topic generation complete! Check the './output' folder for your CSV, PNG, and TXT log.")