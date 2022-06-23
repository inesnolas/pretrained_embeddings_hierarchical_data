import numpy as np
import os
import json
import pandas as pd
# import data_functions as df
# import get_audioset_embeddings as ge
# import manipulate_embeddings



import soundfile as sf
import torch

import hearbaseline.wav2vec2 as wav2vec
import hearbaseline.torchopenl3 as openl3
import hearbaseline.vggish as vggish 
<<<<<<< HEAD
import hearbaseline.vqt as vqt
=======
>>>>>>> c915441c4180111b30c461b2c899ba9286eef2ff

import h5py
import os



<<<<<<< HEAD


=======
# base_folder_save_embeddings = "/homes/in304/extract_embeddings_HEAR_baselines/embeddings"
# dataset_folder = os.path.join(base_folder_save_embeddings, "TUT_ASC2016") 
# open_l3_embeddings_path = os.path.join(dataset_folder, "openL3_embeddings" ) 
# wav2vec_embeddings_path = os.path.join(dataset_folder, "wav2vec_embeddings" ) 
# vggish_embeddings_path = os.path.join(dataset_folder, "vggish_embeddings" ) 

# 0 - created csv at dcase_datafunctions.py
>>>>>>> c915441c4180111b30c461b2c899ba9286eef2ff
# master_csv = '/homes/in304/extract_embeddings_HEAR_baselines/TUT_data_9scenes_3families.csv'

# master_csv = "/homes/in304/extract_embeddings_HEAR_baselines/threeBirdSpecies_data_9individals_3families.csv"

def get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder):


    model_wav2vec = wav2vec.load_model(model_hub = "facebook/wav2vec2-base-100k-voxpopuli") # embedding size =768 , default is facebook/wav2vec2-large-100k-voxpopuli embedding size 1024
    model_openl3_env = openl3.load_model( content_type="env", embedding_size=512) # defaults use content-type music, embedding siez =6400
    model_openl3_music = openl3.load_model( content_type="music", embedding_size=512)

    model_vggish = vggish.load_model()
    dataset = pd.read_csv(dataset_csv)

    # 1 - Compute embeddings 
    # one file at a time so we can work with different lenght of files witout having to pad or manipulate shorter recordings!


    for i in range(len(dataset)):
        if i%100 ==0:
            print("processing file", str(i), " of ", len(dataset))
        wavfilename = dataset.iloc[i].wavfilename
        audio_file_path = os.path.join(audio_folder, wavfilename)
        audio, sr = sf.read(audio_file_path)
        audio_tensor = torch.from_numpy(audio).reshape(1,-1).type(dtype=torch.FloatTensor)
        

        emb_wav2vec2 = wav2vec.get_scene_embeddings(audio_tensor.cuda(), model_wav2vec) 
        emb_openl3_env = openl3.get_scene_embeddings(audio_tensor.cuda(), model_openl3_env) 
        emb_openl3_music = openl3.get_scene_embeddings(audio_tensor.cuda(), model_openl3_music)

        emb_vggish = vggish.get_scene_embeddings(audio_tensor.cuda(), model_vggish) #embedding shape = 128

    
        # save embeddings with filename as wavname
        np.save(os.path.join(output_folder, wavfilename[0:-4]+'_wav2vec'), emb_wav2vec2.detach().cpu().numpy())
        np.save(os.path.join(output_folder, wavfilename[0:-4]+'_openl3_env'), emb_openl3_env.detach().cpu().numpy())
        np.save(os.path.join(output_folder, wavfilename[0:-4]+'_openl3_music'), emb_openl3_music.detach().cpu().numpy())
        np.save(os.path.join(output_folder, wavfilename[0:-4]+'_vggish'), emb_vggish.detach().cpu().numpy())

    return

if __name__ == "__main__" :


    # 3BIRDSPECIES
    
    audio_folder = '/import/c4dm-datasets/animal_identification/AAII_paper_augmented_dataset/AAII_augmented_data'
    output_folder = '/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals'

    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/1200_train.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)


    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/207_test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)

    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/300_val.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)
    
    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/3BirdSpecies9individuals/unseen_test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)
    

    # NSYNTH
    audio_folder = '/import/c4dm-datasets/nsynth/nsynth-train/audio'
    output_folder = '/homes/in304/extract_embeddings_HEAR_baselines/nsynth'

    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/nsynth/1200_train.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)


    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/nsynth/207_test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)

    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/nsynth/300_val.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)
    
    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/nsynth/unseen_test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)


    #TUTasc
    audio_folder = '/import/c4dm-datasets/TUT_acoustic_scenes_2016_dev/TUT-acoustic-scenes-2016-development/audio/'
    output_folder = '/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016'

    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/train.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)


    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)

    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/val.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)
    
    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/TUT_ASC2016/unseen_test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)





   