# from rank_based_loss import evaluate_cluster_quality_based_gt_annotations
import data_functions as df
import pandas as pd
import torch
import numpy as np
import RBL.utils.evaluation as ev
import os
from scipy.spatial import distance
from sklearn import neighbors, datasets
import sklearn.metrics as skmetrics
import sklearn.preprocessing
import csv
from sklearn.decomposition import PCA

def get_initial_embeddings (data_generator, embedding_dimensions, batch_size): # no idea why i need this one anymore...
    
    embeddings_every_epoch = np.empty((batch_size, embedding_dimensions))
    labels_every_epoch=np.asarray([])
    for x , y in data_generator:
        x=x.squeeze()
        embeddings_every_epoch = np.concatenate((embeddings_every_epoch,  x.detach().numpy()), axis = 0) 
        labels_every_epoch = np.concatenate((labels_every_epoch, np.asarray(y)))

    embeddings_every_epoch =  np.delete(embeddings_every_epoch, range(batch_size), 0)
    return embeddings_every_epoch, labels_every_epoch

def normalize_data(embeddings):

    normalized_embeddings, norms = sklearn.preprocessing.normalize(embeddings, return_norm=True)
    return normalized_embeddings

def evaluate_initial_embeddings(dataset, experiments_folder, test_csv, train_csv, val_csv, hard_test_csv , bs, tgt_labels, embeddings_model, embedding_dimensions=128, n_dims=None):
    test_df = pd.read_csv(os.path.join(experiments_folder,test_csv), dtype = str)#'test.csv'), dtype = str)
    train_df = pd.read_csv(os.path.join(experiments_folder,train_csv), dtype = str)
    val_df = pd.read_csv(os.path.join(experiments_folder,val_csv), dtype = str)
    htest_df = pd.read_csv(os.path.join(experiments_folder,hard_test_csv), dtype = str)
   
    initial_embeddings_path = os.path.join(experiments_folder, 'normalized_embeddings')
    test_initial_embeddings_path = os.path.join(initial_embeddings_path, 'test')
    train_initial_embeddings_path = os.path.join(initial_embeddings_path, 'train')
    val_initial_embeddings_path = os.path.join(initial_embeddings_path, 'val')
    htest_initial_embeddings_path = os.path.join(initial_embeddings_path, 'unseen_test')


    test_set = df.HierarchicalLabelsEmbeddings(test_df, test_initial_embeddings_path, embeddings_model, tgt_labels)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False)
    len_test = len(test_set)
    htest_set = df.HierarchicalLabelsEmbeddings(htest_df, htest_initial_embeddings_path, embeddings_model,tgt_labels)
    htest_generator = torch.utils.data.DataLoader(htest_set, batch_size=bs, shuffle=False)
    len_htest = len(htest_set)
    training_set = df.HierarchicalLabelsEmbeddings(train_df, train_initial_embeddings_path, embeddings_model,tgt_labels)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=bs, shuffle=False)
    len_train = len(training_set)
    validation_set = df.HierarchicalLabelsEmbeddings(val_df , val_initial_embeddings_path, embeddings_model,tgt_labels)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=bs, shuffle=False)
    len_val = len(validation_set)


    embeddings_train, gt__train_labels = get_initial_embeddings (training_generator, embedding_dimensions, bs)
    embeddings_val, gt__val_labels = get_initial_embeddings (validation_generator, embedding_dimensions, bs)
    embeddings_test, gt__test_labels = get_initial_embeddings (test_generator, embedding_dimensions, bs)
    embeddings_htest, gt__htest_labels = get_initial_embeddings (htest_generator, embedding_dimensions, bs)


     ## PCA transformation of initial embeddings!

    if n_dims == None:
        n_dims = embeddings_val.shape[1]  # original dimensions(128?) just for printing ORiginal dimesnions do not perform PCA
    elif n_dims > embeddings_train.shape[0]:
        return dataset, n_dims, '', '', '', '', '', '', '', '', '', '', '', ''
    elif (n_dims<embeddings_train.shape[0]):
        # pca = pca_initial_embeddings_fit(embeddings_train, n_dims)
        pca = PCA(n_components=n_dims)
        pca.fit(embeddings_train)

        embeddings_train = pca.transform(embeddings_train)
        embeddings_val = pca.transform(embeddings_val)
        embeddings_test = pca.transform(embeddings_test)
        embeddings_htest = pca.transform(embeddings_htest)
    
    embeddings_train = normalize_data(embeddings_train) # transform to norm 1
    embeddings_val = normalize_data(embeddings_val)
    embeddings_test = normalize_data(embeddings_test)
    embeddings_htest = normalize_data(embeddings_htest)



    train_silhouette_score_finelevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_train, gt__train_labels)
    labels_coarse_train = [l.split('_')[1] for l in gt__train_labels]
    train_silhouette_score_2ndlevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_train, labels_coarse_train)

       
    val_silhouette_score_finelevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_val, gt__val_labels)
    labels_coarse_val = [l.split('_')[1] for l in gt__val_labels]
    val_silhouette_score_2ndlevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_val, labels_coarse_val)


    t_silhouette_score_finelevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_test, gt__test_labels)
    labels_coarse_test = [l.split('_')[1] for l in gt__test_labels]
    t_silhouette_score_2ndlevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_test, labels_coarse_test)


    hht_silhouette_score_finelevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_htest, gt__htest_labels)
    labels_coarse_htest = [l.split('_')[1] for l in gt__htest_labels]
    hht_silhouette_score_2ndlevel_labels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_htest, labels_coarse_htest)

    val_acc_lvl0_by_k = []
    val_acc_lvl1_by_k = []
    for k in range(40):
        k = k+1
        clf = neighbors.KNeighborsClassifier(k, weights='distance', algorithm='brute', metric='cosine')
        clf.fit(embeddings_train, gt__train_labels)

        predictions_val = clf.predict(embeddings_val)
        
        acc_lvl0 = skmetrics.accuracy_score(gt__val_labels, predictions_val, normalize=True, sample_weight=None)
        gt__val_labels_lvl1 = [l.split('_')[1] for l in gt__val_labels]
        predictions_lvl1 = [p.split('_')[1] for p in predictions_val]
        acc_lvl1 = skmetrics.accuracy_score(gt__val_labels_lvl1, predictions_lvl1, normalize=True, sample_weight=None)

    
        # print('k = ', k, 'average_accuracies = ', (acc_lvl0)
        val_acc_lvl0_by_k.append(acc_lvl0)
        val_acc_lvl1_by_k.append(acc_lvl1)

    max_value_lvl0 = max(val_acc_lvl0_by_k)
    k_best_lvl0 = val_acc_lvl0_by_k.index(max_value_lvl0) + 1

    max_value_lvl1 = max(val_acc_lvl1_by_k)
    k_best_lvl1 = val_acc_lvl1_by_k.index(max_value_lvl1) + 1

    clf_lvl0 = neighbors.KNeighborsClassifier(k_best_lvl0, weights='distance', algorithm='brute', metric='cosine')
    clf_lvl0.fit(embeddings_train, gt__train_labels)
    predictions = clf_lvl0.predict(embeddings_test)
    acc_lvl0 = skmetrics.accuracy_score(gt__test_labels, predictions, normalize=True, sample_weight=None)

    clf_lvl1 = neighbors.KNeighborsClassifier(k_best_lvl1, weights='distance', algorithm='brute', metric='cosine')
    clf_lvl1.fit(embeddings_train, gt__train_labels)

    predictions_test_lvl1 = clf_lvl1.predict(embeddings_test)
    gt_test_labels_lvl1 = [l.split('_')[1] for l in gt__test_labels]
    predictions_lvl1 = [p.split('_')[1] for p in predictions_test_lvl1]
    acc_lvl1 = skmetrics.accuracy_score(gt_test_labels_lvl1, predictions_lvl1, normalize=True, sample_weight=None)

    predictions_htest = clf_lvl0.predict(embeddings_htest) 
    acc_lvl0_htest = skmetrics.accuracy_score(gt__htest_labels, predictions_htest, normalize=True, sample_weight=None)

    predictions_htest_lvl1 = clf_lvl1.predict(embeddings_htest)
    gt_htest_labels_lvl1 = [l.split('_')[1] for l in gt__htest_labels]
    predictions_hlvl1 = [p.split('_')[1] for p in predictions_htest_lvl1]
    acc_lvl1_htest = skmetrics.accuracy_score(gt_htest_labels_lvl1, predictions_hlvl1, normalize=True, sample_weight=None)

    return dataset, n_dims, t_silhouette_score_finelevel_labels, t_silhouette_score_2ndlevel_labels, (t_silhouette_score_finelevel_labels +t_silhouette_score_2ndlevel_labels)/2 , acc_lvl0, k_best_lvl0, acc_lvl1, k_best_lvl1, hht_silhouette_score_finelevel_labels, hht_silhouette_score_2ndlevel_labels, (hht_silhouette_score_finelevel_labels +hht_silhouette_score_2ndlevel_labels)/2, acc_lvl0_htest, acc_lvl1_htest
    


def create_results_table(emb_model, emb_dim):
    columns = ["Dataset", "dims", "Sil_fine", "Sil_coarse", "Sil_avg", "KNN_acc_fine", "n_k_fine", "KNN_acc_coarse", "n_k_coarse", "HSil_fine", "HSil_coarse", "HSil_avg", "HKNN_acc_fine", "HKNN_acc_coarse"]

    dims = 2**np.arange(2, 10)
    with open('Initial_embeddings_'+emb_model+'_evaluation_results.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(columns)

        # writer.writerow(evaluate_initial_embeddings(dataset, experiments_folder, test_csv, train_csv, val_csv, hard_test_csv , bs, target_labels, embeddings_model, embedding_dimensions=128, n_dims=None)
        writer.writerow(evaluate_initial_embeddings("3BirdSpecies9individuals","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals", "207_test.csv", "1200_train.csv", "300_val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels', embeddings_model= emb_model, embedding_dimensions=emb_dim, n_dims=3))
        for i in dims:
            if i>= emb_dim:
                break
            else:
                writer.writerow(evaluate_initial_embeddings("3BirdSpecies9individuals","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals", "207_test.csv", "1200_train.csv", "300_val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels', embeddings_model= emb_model,embedding_dimensions=emb_dim, n_dims=i ))
        writer.writerow(evaluate_initial_embeddings("3BirdSpecies9individuals","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals", "207_test.csv", "1200_train.csv", "300_val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels',embeddings_model= emb_model,embedding_dimensions=emb_dim))
        
        print('\n Nsynth data')
        writer.writerow(evaluate_initial_embeddings("Nsynth","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth", "207_test.csv", "1200_train.csv", "300_val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels', embeddings_model= emb_model,embedding_dimensions=emb_dim, n_dims=3))
        for i in dims:
            if i>= emb_dim:
                break
            else:
                writer.writerow(evaluate_initial_embeddings("Nsynth","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth", "207_test.csv", "1200_train.csv", "300_val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels',embeddings_model= emb_model,embedding_dimensions=emb_dim, n_dims=i ))
        writer.writerow(evaluate_initial_embeddings("Nsynth","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth", "207_test.csv", "1200_train.csv", "300_val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels', embeddings_model= emb_model,embedding_dimensions=emb_dim))
        
        print('\n TuTASC data')
        writer.writerow(evaluate_initial_embeddings("TUTasc","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016", "test.csv", "train.csv", "val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels', embeddings_model= emb_model,embedding_dimensions=emb_dim, n_dims=3))
        for i in dims:
            if i>= emb_dim:
                break
            else:
                writer.writerow(evaluate_initial_embeddings("TUTasc","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016", "test.csv", "train.csv", "val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels',embeddings_model= emb_model,embedding_dimensions=emb_dim, n_dims=i ))
        writer.writerow(evaluate_initial_embeddings("TUTasc","/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016", "test.csv", "train.csv", "val.csv", "unseen_test.csv" , bs=15, tgt_labels = 'hierarchical_labels', embeddings_model= emb_model,embedding_dimensions=emb_dim))
        
    return



    

if __name__ == "__main__":


    create_results_table('vggish', 128)
    create_results_table('wav2vec',768 )
    create_results_table('openl3_music', 512)
    create_results_table('openl3_env', 512)




    