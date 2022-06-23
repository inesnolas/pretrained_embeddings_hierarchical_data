import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class HierarchicalLabelsEmbeddings(Dataset):
    def __init__(self, partition_dataframe, features_folder,  embeddings_model='vggish', target_labels='hierarchical_labels'): #['vggish', 'wav2vec', 'openl3_music', 'openl3_env']
        self.dataset = partition_dataframe 
        self.wavs_list = list(self.dataset.wavfilename)
        self.features_folder = features_folder
        self.target_labels = target_labels
        self.labels = list(self.dataset[target_labels])
        self.emb_model= embeddings_model
        # self.target_label = level_label
        # self.label_values = labels_dict
        # self.segment_size_sec = segment_size_sec

    def __len__(self):
        return len(self.wavs_list)


    def __getitem__(self,idx):
        
        ID = self.wavs_list[idx][0:-4]+'_'+self.emb_model+'.npy'
        X = np.load(os.path.join(self.features_folder, ID))
        y = self.labels[idx]
        return X, y

def normalize_embeddings_based_training_set(dataset_csv, outfolder, raw_embeddings_folder, mean_embeddings, std_embeddings, embeddings_model=['vggish', 'wav2vec', 'openl3_music', 'openl3_env'], set_name='train'):
    
    data = pd.read_csv(dataset_csv)


    #compute stats matrices!


    for m, emb_model in enumerate(embeddings_model):    
        embeddings = []
        filenames = []
        for i in range(len(data.index)):
            filename = data.iloc[i].wavfilename[0:-4] + '_'+emb_model+'.npy'
            filenames.append(filename)
            embedding = np.load(os.path.join(raw_embeddings_folder, filename))
            embeddings.append(embedding)    
        
        embeddings = np.asarray(embeddings)


        if set_name =='train':
            mean_embeddings_bymodel = np.mean(embeddings,0)
            std_embeddings_bymodel = np.std(embeddings,0) + 0.0001
            np.save(os.path.join(outfolder, 'mean_embeddings_'+emb_model+'_train.npy'), mean_embeddings_bymodel)
            np.save(os.path.join(outfolder, 'std_embeddings_'+emb_model+'_train.npy'), std_embeddings_bymodel)
            normalized_embeddings =  (embeddings - mean_embeddings_bymodel)/std_embeddings_bymodel
        
        
        else:
            normalized_embeddings =  (embeddings - mean_embeddings[m])/std_embeddings[m]

        if not os.path.exists(os.path.join(outfolder, set_name)):
            os.mkdir(os.path.join(outfolder, set_name))

        for e in range(len(filenames)):
            np.save(os.path.join(outfolder, set_name, filenames[e]), normalized_embeddings[e])
    
    return 




if __name__ == "__main__" :


    # save_folder = "/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/normalized_embeddings"

    # # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/1200_train.csv', 
    # #                                                 outfolder=save_folder, 
    # #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/raw_embeddings",
    # #                                                 mean_embeddings=[], std_embeddings=[], set_name='train')


    # mean_embeddings_wav2vec = np.load(os.path.join(save_folder, 'mean_embeddings_wav2vec_train.npy'))
    # std_embeddings_wav2vec = np.load(os.path.join(save_folder, 'std_embeddings_wav2vec_train.npy'))
    # mean_embeddings_vggish = np.load(os.path.join(save_folder, 'mean_embeddings_vggish_train.npy'))
    # std_embeddings_vggish = np.load(os.path.join(save_folder, 'std_embeddings_vggish_train.npy'))
    # mean_embeddings_openl3_env = np.load(os.path.join(save_folder, 'mean_embeddings_openl3_env_train.npy'))
    # std_embeddings_openl3_env = np.load(os.path.join(save_folder, 'std_embeddings_openl3_env_train.npy'))
    # mean_embeddings_openl3_music = np.load(os.path.join(save_folder, 'mean_embeddings_openl3_music_train.npy'))
    # std_embeddings_openl3_music = np.load(os.path.join(save_folder, 'std_embeddings_openl3_music_train.npy'))                                                 

    # #follow same order as in embeddings_model = ['vggish', 'wav2vec', 'openl3_music', 'openl3_env']
    # mean_list = [mean_embeddings_vggish, mean_embeddings_wav2vec, mean_embeddings_openl3_music, mean_embeddings_openl3_env]
    # std_list= [std_embeddings_vggish, std_embeddings_wav2vec, std_embeddings_openl3_music, std_embeddings_openl3_env]

    # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/300_val.csv', 
    #                                                 outfolder=save_folder, 
    #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines//3BirdSpecies9individuals/raw_embeddings",
    #                                                 mean_embeddings=mean_list, std_embeddings=std_list, set_name='val')
    # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines//3BirdSpecies9individuals/207_test.csv', 
    #                                                 outfolder=save_folder, 
    #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/raw_embeddings",
    #                                                 mean_embeddings=mean_list, std_embeddings=std_list, set_name='test')
    # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/unseen_test.csv', 
    #                                                 outfolder=save_folder, 
    #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/raw_embeddings",
    #                                                 mean_embeddings=mean_list, std_embeddings=std_list, set_name='unseen_test')

# NSYNTH

    # save_folder = "/homes/in304/extract_embeddings_HEAR_baselines/nsynth/normalized_embeddings"

    # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/nsynth/1200_train.csv', 
    #                                                 outfolder=save_folder, 
    #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/nsynth/raw_embeddings",
    #                                                 mean_embeddings=[], std_embeddings=[], set_name='train')


    # mean_embeddings_wav2vec = np.load(os.path.join(save_folder, 'mean_embeddings_wav2vec_train.npy'))
    # std_embeddings_wav2vec = np.load(os.path.join(save_folder, 'std_embeddings_wav2vec_train.npy'))
    # mean_embeddings_vggish = np.load(os.path.join(save_folder, 'mean_embeddings_vggish_train.npy'))
    # std_embeddings_vggish = np.load(os.path.join(save_folder, 'std_embeddings_vggish_train.npy'))
    # mean_embeddings_openl3_env = np.load(os.path.join(save_folder, 'mean_embeddings_openl3_env_train.npy'))
    # std_embeddings_openl3_env = np.load(os.path.join(save_folder, 'std_embeddings_openl3_env_train.npy'))
    # mean_embeddings_openl3_music = np.load(os.path.join(save_folder, 'mean_embeddings_openl3_music_train.npy'))
    # std_embeddings_openl3_music = np.load(os.path.join(save_folder, 'std_embeddings_openl3_music_train.npy'))                                                 

    # #follow same order as in embeddings_model = ['vggish', 'wav2vec', 'openl3_music', 'openl3_env']
    # mean_list = [mean_embeddings_vggish, mean_embeddings_wav2vec, mean_embeddings_openl3_music, mean_embeddings_openl3_env]
    # std_list= [std_embeddings_vggish, std_embeddings_wav2vec, std_embeddings_openl3_music, std_embeddings_openl3_env]

    # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/nsynth/300_val.csv', 
    #                                                 outfolder=save_folder, 
    #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/nsynth/raw_embeddings",
    #                                                 mean_embeddings=mean_list, std_embeddings=std_list, set_name='val')
    # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/nsynth/207_test.csv', 
    #                                                 outfolder=save_folder, 
    #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/nsynth/raw_embeddings",
    #                                                 mean_embeddings=mean_list, std_embeddings=std_list, set_name='test')
    # normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/nsynth/unseen_test.csv', 
    #                                                 outfolder=save_folder, 
    #                                                 raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/nsynth/raw_embeddings",
    #                                                 mean_embeddings=mean_list, std_embeddings=std_list, set_name='unseen_test')

# TUT asc

    save_folder = "/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/normalized_embeddings"

    normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/train.csv', 
                                                    outfolder=save_folder, 
                                                    raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/raw_embeddings",
                                                    mean_embeddings=[], std_embeddings=[], set_name='train')


    mean_embeddings_wav2vec = np.load(os.path.join(save_folder, 'mean_embeddings_wav2vec_train.npy'))
    std_embeddings_wav2vec = np.load(os.path.join(save_folder, 'std_embeddings_wav2vec_train.npy'))
    mean_embeddings_vggish = np.load(os.path.join(save_folder, 'mean_embeddings_vggish_train.npy'))
    std_embeddings_vggish = np.load(os.path.join(save_folder, 'std_embeddings_vggish_train.npy'))
    mean_embeddings_openl3_env = np.load(os.path.join(save_folder, 'mean_embeddings_openl3_env_train.npy'))
    std_embeddings_openl3_env = np.load(os.path.join(save_folder, 'std_embeddings_openl3_env_train.npy'))
    mean_embeddings_openl3_music = np.load(os.path.join(save_folder, 'mean_embeddings_openl3_music_train.npy'))
    std_embeddings_openl3_music = np.load(os.path.join(save_folder, 'std_embeddings_openl3_music_train.npy'))                                                 

    #follow same order as in embeddings_model = ['vggish', 'wav2vec', 'openl3_music', 'openl3_env']
    mean_list = [mean_embeddings_vggish, mean_embeddings_wav2vec, mean_embeddings_openl3_music, mean_embeddings_openl3_env]
    std_list= [std_embeddings_vggish, std_embeddings_wav2vec, std_embeddings_openl3_music, std_embeddings_openl3_env]

    normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/val.csv', 
                                                    outfolder=save_folder, 
                                                    raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/raw_embeddings",
                                                    mean_embeddings=mean_list, std_embeddings=std_list, set_name='val')
    normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/test.csv', 
                                                    outfolder=save_folder, 
                                                    raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/raw_embeddings",
                                                    mean_embeddings=mean_list, std_embeddings=std_list, set_name='test')
    normalize_embeddings_based_training_set('/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/unseen_test.csv', 
                                                    outfolder=save_folder, 
                                                    raw_embeddings_folder="/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/raw_embeddings",
                                                    mean_embeddings=mean_list, std_embeddings=std_list, set_name='unseen_test')


