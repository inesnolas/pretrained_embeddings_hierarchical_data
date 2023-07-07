from RBL.models import SingleLayer_net as single_layer
from RBL.loss_functions import rank_based_loss as rbl
from RBL.loss_functions import quadruplet_loss as quadl
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_functions as df
import numpy as np
import os
from sklearn.metrics import confusion_matrix
# import evaluation as ev
import json
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_with_RBL( pretrained_model_name, dataset_name, exp_name, exp_folder, train_csv, val_csv, initial_embeddings_size, 
        output_embeddings_size, early_stopping_patience=20,
        learning_rate=1e-5, batch_size=12, max_n_epochs=20000,
        save_training_embeddings_to_plot = True,
        shuffle = False,  
        drop_last = False  ):

        wandb.init(project='pretrained_embeddings_from_HEAR')
        wandb.run.name = exp_name

        # Flag_compute_cluster_scores = True  #???
        # data_sets_csv_folder = exp_folder

        # if standardized_data:
        initial_embeddings_path = os.path.join(exp_folder, 'normalized_embeddings')
        train_initial_embeddings_path = os.path.join(initial_embeddings_path, 'train')
        val_initial_embeddings_path = os.path.join(initial_embeddings_path, 'val')


        results_folder = os.path.join(exp_folder, "results_"+exp_name)
        checkpoints_folder = os.path.join(results_folder, "checkpoints")
        if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)

        if save_training_embeddings_to_plot:
                if not os.path.exists(os.path.join(checkpoints_folder, "Embeddings2plot")):
                        os.mkdir(os.path.join(checkpoints_folder, "Embeddings2plot"))
        
        train_df = pd.read_csv(os.path.join(exp_folder, train_csv), dtype = str)
        val_df = pd.read_csv(os.path.join(exp_folder, val_csv), dtype = str)


        configs = {"EMBEDDINGS_SIZE" : initial_embeddings_size,
        "output_EMBEDDINGS_SIZE" : output_embeddings_size, 
        "EARLY_STOPPING_PTC" : early_stopping_patience, #20
        "LR" : learning_rate, #1e-5,
        "BATCH_SIZE" : batch_size, # 12,
        "n_epochs" : max_n_epochs, #20000, 
        }
        params = {'batch_size': configs["BATCH_SIZE"],'shuffle': shuffle, 'drop_last': drop_last}


        training_set = df.HierarchicalLabelsEmbeddings(train_df, train_initial_embeddings_path, pretrained_model_name, target_labels='hierarchical_labels')#,'species','taxon'])
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        len_train = len(training_set)


        validation_set = df.HierarchicalLabelsEmbeddings(val_df , val_initial_embeddings_path, pretrained_model_name, target_labels='hierarchical_labels')#,'species','taxon'])
        params_val = {'batch_size': configs["BATCH_SIZE"],'shuffle': False, 'drop_last': False}
        validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
        len_val = len(validation_set)

        model =single_layer.SingleLayerHypersphereConstraint(configs)

        wandb.watch(model)
        wandb.config = configs
        wandb.config["architecture"] = "SingleLayerHypersphereConstraint"
        wandb.config["dataset"] = dataset_name
        with open(os.path.join(results_folder, 'configs_dict'), "w") as c:
                json.dump(configs, c)

        checkpoint_name = rbl.train_RbL(model, training_generator, validation_generator,
                                        checkpoints_folder, configs['EARLY_STOPPING_PTC'], save_training_embeddings_to_plot, 
                                        configs['n_epochs'], configs, distance='cosine',
                                        number_of_ranks = 4)
        wandb.finish()                       
        return checkpoint_name




def train_with_QUADloss( pretrained_model_name, dataset_name, exp_name, exp_folder, train_csv, val_csv, initial_embeddings_size, 
        output_embeddings_size, early_stopping_patience=20,
        learning_rate=1e-5, batch_size=3, max_n_epochs=20000,
        save_training_embeddings_to_plot = True,
        shuffle = False,  
        drop_last = False,
        margin_alpha =0.2, margin_beta=0.1 ):

        wandb.init(project='pretrained_embeddings_from_HEAR')
        wandb.run.name = exp_name

        Flag_compute_cluster_scores = True
        data_sets_csv_folder = exp_folder

        initial_embeddings_path = os.path.join(exp_folder, 'normalized_embeddings')
        train_initial_embeddings_path = os.path.join(initial_embeddings_path, 'train')
        val_initial_embeddings_path = os.path.join(initial_embeddings_path, 'val')


        results_folder = os.path.join(exp_folder, "results_"+exp_name)
        checkpoints_folder = os.path.join(results_folder, "checkpoints")
        if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)

        if save_training_embeddings_to_plot:
                if not os.path.exists(os.path.join(checkpoints_folder, "Embeddings2plot")):
                        os.mkdir(os.path.join(checkpoints_folder, "Embeddings2plot"))
        
        train_df = pd.read_csv(os.path.join(exp_folder, train_csv), dtype = str)
        val_df = pd.read_csv(os.path.join(exp_folder, val_csv), dtype = str)


        configs = {"EMBEDDINGS_SIZE" : initial_embeddings_size,
        "output_EMBEDDINGS_SIZE" : output_embeddings_size, 
        "EARLY_STOPPING_PTC" : early_stopping_patience, #20
        "LR" : learning_rate, #1e-5,
        "BATCH_SIZE" : batch_size, # 12,
        "n_epochs" : max_n_epochs, #20000, 
        }
        params = {'batch_size': configs["BATCH_SIZE"],'shuffle': shuffle, 'drop_last': drop_last}


        training_set = df.QuadrupletsHierarchicalLabelsEmbeddings(train_df, train_initial_embeddings_path, pretrained_model_name)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        len_train = len(training_set)


        validation_set = df.QuadrupletsHierarchicalLabelsEmbeddings(val_df , val_initial_embeddings_path, pretrained_model_name)
        params_val = {'batch_size': configs["BATCH_SIZE"],'shuffle': False, 'drop_last': False}
        validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
        len_val = len(validation_set)




        model =single_layer.SingleLayerHypersphereConstraint(configs)

        wandb.watch(model)
        wandb.config = configs
        wandb.config["architecture"] = "SingleLayerHypersphereConstraint"
        wandb.config["dataset"] = dataset_name
        with open(os.path.join(results_folder, 'configs_dict'), "w") as c:
                json.dump(configs, c)

        checkpoint_name = quadl.train_quadL(model,margin_alpha, margin_beta, training_generator, validation_generator, checkpoints_folder,
                                 early_stopping_patience, save_training_embeddings_to_plot, configs['n_epochs'], configs, distance='cosine')

        wandb.finish()
        return checkpoint_name        

if __name__ == "__main__":

        

        pretrained_model_names_list = ['BirdNET_1024']#['crepe','vggish', 'openl3_env', 'openl3_music', 'wav2vec']
        pretrained_model_dims = [1024] #[2048 128, 512, 512, 768]
        output_embeddings_size = 3 
        # 3 bird data
        
        dataset_name = '3BirdSpecies9individuals'
        exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals"

        for i, pretrained_model_name in enumerate(pretrained_model_names_list):
                initial_embeddings_size = pretrained_model_dims[i]
                
                exp_name = '3BirsSpecies_'+pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_RBL_'+str(output_embeddings_size)

                train_with_RBL(pretrained_model_name, dataset_name, exp_name, exp_folder, 
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/1200_train.csv",
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300_val.csv", 
                        initial_embeddings_size, 
                        output_embeddings_size, early_stopping_patience=20,
                        learning_rate=1e-5, batch_size=12, max_n_epochs=20000,
                        save_training_embeddings_to_plot = True,
                        shuffle = False,  
                        drop_last = False)

                exp_name = '3BirsSpecies_'+ pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_QUADL_'+str(output_embeddings_size)

                train_with_QUADloss(pretrained_model_name, dataset_name, exp_name, exp_folder,
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300quadruplets_train.csv", 
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/75quadruplets_val.csv",
                        initial_embeddings_size, 
                        output_embeddings_size, early_stopping_patience=20,
                        learning_rate=1e-5, batch_size=3, max_n_epochs=20000,
                        save_training_embeddings_to_plot = True,
                        shuffle = False,  
                        drop_last = False,
                        margin_alpha =0.2, margin_beta=0.1)

        #         # print('stop')

        # #NSYNTH


        dataset_name = 'nsynth'
        exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth"

        for i, pretrained_model_name in enumerate(pretrained_model_names_list):
                initial_embeddings_size = pretrained_model_dims[i]
                
                exp_name = dataset_name+'_'+pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_RBL_'+str(output_embeddings_size)

                train_with_RBL(pretrained_model_name, dataset_name, exp_name, exp_folder, 
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth/1200_train.csv",
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth/300_val.csv", 
                        initial_embeddings_size, 
                        output_embeddings_size, early_stopping_patience=20,
                        learning_rate=1e-5, batch_size=12, max_n_epochs=20000,
                        save_training_embeddings_to_plot = True,
                        shuffle = False,  
                        drop_last = False)

                exp_name = dataset_name+'_'+ pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_QUADL_'+str(output_embeddings_size)

                train_with_QUADloss(pretrained_model_name, dataset_name, exp_name, exp_folder,
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth/300quadruplets_train.csv", 
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/nsynth/75quadruplets_val.csv",
                        initial_embeddings_size, 
                        output_embeddings_size, early_stopping_patience=20,
                        learning_rate=1e-5, batch_size=3, max_n_epochs=20000,
                        save_training_embeddings_to_plot = True,
                        shuffle = False,  
                        drop_last = False,
                        margin_alpha =0.2, margin_beta=0.1)


        # #TUT asc
        dataset_name = 'TUT_ASC2016'
        exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016"

        for i, pretrained_model_name in enumerate(pretrained_model_names_list):
                initial_embeddings_size = pretrained_model_dims[i]
                
                exp_name = dataset_name+'_'+pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_RBL_'+str(output_embeddings_size)

                train_with_RBL(pretrained_model_name, dataset_name, exp_name, exp_folder, 
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016/train.csv",
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016/val.csv", 
                        initial_embeddings_size, 
                        output_embeddings_size, early_stopping_patience=20,
                        learning_rate=1e-5, batch_size=12, max_n_epochs=20000,
                        save_training_embeddings_to_plot = True,
                        shuffle = False,  
                        drop_last = False)

                exp_name = dataset_name+'_'+ pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_QUADL_'+str(output_embeddings_size)

                train_with_QUADloss(pretrained_model_name, dataset_name, exp_name, exp_folder,
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016/train_quadruplets.csv", 
                        "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/TUT_ASC2016/val_quadruplets.csv",
                        initial_embeddings_size, 
                        output_embeddings_size, early_stopping_patience=20,
                        learning_rate=1e-5, batch_size=3, max_n_epochs=20000,
                        save_training_embeddings_to_plot = True,
                        shuffle = False,  
                        drop_last = False,
                        margin_alpha =0.2, margin_beta=0.1)

print('stop')

##########################################################################################################
        #COMPUTE larger dimension embedding space with RBL ? (128)

        # output_embeddings_size = 128 
        # dataset_name = '3BirdSpecies9individuals'
        # exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals"

        # initial_embeddings_size = 128
        # pretrained_model_name = 'vggish'
                
                
        # exp_name = '3BirsSpecies_'+pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_RBL_'+str(output_embeddings_size)

        # train_with_RBL(pretrained_model_name, dataset_name, exp_name, exp_folder, 
        #         "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/1200_train.csv",
        #         "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300_val.csv", 
        #         initial_embeddings_size, 
        #         output_embeddings_size, early_stopping_patience=20,
        #         learning_rate=1e-5, batch_size=12, max_n_epochs=20000,
        #         save_training_embeddings_to_plot = True,
        #         shuffle = False,  
        #         drop_last = False)


        # dataset_name = '3BirdSpecies9individuals'
        # exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals"
        # output_embeddings_size = 128  
        # pretrained_models = [ 'openl3_env', 'openl3_music', 'wav2vec']
        # pretrained_model_dims = [512, 512, 768]
        # for i, pretrained_model_name in enumerate(pretrained_models):
        #         initial_embeddings_size = pretrained_model_dims[i]
                        
        #         exp_name = '3BirsSpecies_'+pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_RBL_'+str(output_embeddings_size)

        #         train_with_RBL(pretrained_model_name, dataset_name, exp_name, exp_folder, 
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/1200_train.csv",
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300_val.csv", 
        #                 initial_embeddings_size, 
        #                 output_embeddings_size, early_stopping_patience=20,
        #                 learning_rate=1e-5, batch_size=12, max_n_epochs=20000,
        #                 save_training_embeddings_to_plot = True,
        #                 shuffle = False,  
        #                 drop_last = False)

        # dataset_name = '3BirdSpecies9individuals'
        # exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals"
        # pretrained_models = [ 'openl3_env', 'openl3_music', 'wav2vec']
        # pretrained_model_dims = [512, 512, 768]
        # for i, pretrained_model_name in enumerate(pretrained_models):
        #         initial_embeddings_size = pretrained_model_dims[i]
        #         output_embeddings_size = initial_embeddings_size          
        #         exp_name = '3BirsSpecies_'+pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_RBL_'+str(output_embeddings_size)

        #         train_with_RBL(pretrained_model_name, dataset_name, exp_name, exp_folder, 
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/1200_train.csv",
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300_val.csv", 
        #                 initial_embeddings_size, 
        #                 output_embeddings_size, early_stopping_patience=20,
        #                 learning_rate=1e-5, batch_size=12, max_n_epochs=20000,
        #                 save_training_embeddings_to_plot = True,
        #                 shuffle = False,  
        #                 drop_last = False)
        
        # #QUADLOSS
        # output_embeddings_size = 128 
        # dataset_name = '3BirdSpecies9individuals'
        # exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals"

        # initial_embeddings_size = 128
        # pretrained_model_name = 'vggish'
                
                
        # exp_name = '3BirsSpecies_'+ pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_QUADL_'+str(output_embeddings_size)

        # train_with_QUADloss(pretrained_model_name, dataset_name, exp_name, exp_folder,
        #         "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300quadruplets_train.csv", 
        #         "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/75quadruplets_val.csv",
        #         initial_embeddings_size, 
        #         output_embeddings_size, early_stopping_patience=20,
        #         learning_rate=1e-5, batch_size=3, max_n_epochs=20000,
        #         save_training_embeddings_to_plot = True,
        #         shuffle = False,  
        #         drop_last = False,
        #         margin_alpha =0.2, margin_beta=0.1)


        # dataset_name = '3BirdSpecies9individuals'
        # exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals"
        # output_embeddings_size = 128  
        # pretrained_models = [ 'vggish','openl3_env', 'openl3_music', 'wav2vec']
        # pretrained_model_dims = [128, 512, 512, 768]
        # for i, pretrained_model_name in enumerate(pretrained_models):
        #         initial_embeddings_size = pretrained_model_dims[i]
                        
                                
        #         exp_name = '3BirsSpecies_'+ pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_QUADL_'+str(output_embeddings_size)

        #         train_with_QUADloss(pretrained_model_name, dataset_name, exp_name, exp_folder,
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300quadruplets_train.csv", 
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/75quadruplets_val.csv",
        #                 initial_embeddings_size, 
        #                 output_embeddings_size, early_stopping_patience=20,
        #                 learning_rate=1e-5, batch_size=3, max_n_epochs=20000,
        #                 save_training_embeddings_to_plot = True,
        #                 shuffle = False,  
        #                 drop_last = False,
        #                 margin_alpha =0.2, margin_beta=0.1)

        # dataset_name = '3BirdSpecies9individuals'
        # exp_folder = "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals"
        # pretrained_models = ['vggish', 'openl3_env', 'openl3_music', 'wav2vec']
        # pretrained_model_dims = [128,512, 512, 768]
        # for i, pretrained_model_name in enumerate(pretrained_models):
        #         initial_embeddings_size = pretrained_model_dims[i]
        #         output_embeddings_size = initial_embeddings_size          
                
        #         exp_name = '3BirsSpecies_'+ pretrained_model_name+'_'+str(initial_embeddings_size)+'dims_'+'_QUADL_'+str(output_embeddings_size)

        #         train_with_QUADloss(pretrained_model_name, dataset_name, exp_name, exp_folder,
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/300quadruplets_train.csv", 
        #                 "/homes/in304/extract_pretrained_embeddings_from_HEARbaselines/3BirdSpecies9individuals/75quadruplets_val.csv",
        #                 initial_embeddings_size, 
        #                 output_embeddings_size, early_stopping_patience=20,
        #                 learning_rate=1e-5, batch_size=3, max_n_epochs=20000,
        #                 save_training_embeddings_to_plot = True,
        #                 shuffle = False,  
        #                 drop_last = False,
        #                 margin_alpha =0.2, margin_beta=0.1)    