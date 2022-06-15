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

import h5py
import os



# base_folder_save_embeddings = "/homes/in304/extract_embeddings_HEAR_baselines/embeddings"
# dataset_folder = os.path.join(base_folder_save_embeddings, "TUT_ASC2016") 
# open_l3_embeddings_path = os.path.join(dataset_folder, "openL3_embeddings" ) 
# wav2vec_embeddings_path = os.path.join(dataset_folder, "wav2vec_embeddings" ) 
# vggish_embeddings_path = os.path.join(dataset_folder, "vggish_embeddings" ) 

# 0 - created csv at dcase_datafunctions.py
# master_csv = '/homes/in304/extract_embeddings_HEAR_baselines/TUT_data_9scenes_3families.csv'

# master_csv = "/homes/in304/extract_embeddings_HEAR_baselines/threeBirdSpecies_data_9individals_3families.csv"

def get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder):

    model_wav2vec = wav2vec.load_model()
    model_openl3 = openl3.load_model()
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
        
        emb_wav2vec2 = wav2vec.get_scene_embeddings(audio_tensor.cuda(), model_wav2vec) #embedding shape = 1024
        emb_openl3 = openl3.get_scene_embeddings(audio_tensor.cuda(), model_openl3) #embeddings shpae = 6144
        emb_vggish = vggish.get_scene_embeddings(audio_tensor.cuda(), model_vggish) #embedding shape = 128

    
        # save embeddings with filename as wavname
        np.save(os.path.join(output_folder, wavfilename[0:-4]+'_wav2vec'), emb_wav2vec2.detach().cpu().numpy())
        np.save(os.path.join(output_folder, wavfilename[0:-4]+'_openl3'), emb_openl3.detach().cpu().numpy())
        np.save(os.path.join(output_folder, wavfilename[0:-4]+'_vggish'), emb_vggish.detach().cpu().numpy())

    return

if __name__ == "__main__" :

    # dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/embeddings/3BirdSpecies9individuals/1200_train.csv"
    audio_folder = '/import/c4dm-datasets/animal_identification/AAII_paper_augmented_dataset/AAII_augmented_data'
    output_folder = '/homes/in304/extract_embeddings_HEAR_baselines/embeddings/3BirdSpecies9individuals'
    # get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)


    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/embeddings/3BirdSpecies9individuals/207_test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)

    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/embeddings/3BirdSpecies9individuals/300_val.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)
    
    dataset_csv = "/homes/in304/extract_embeddings_HEAR_baselines/embeddings/3BirdSpecies9individuals/unseen_test.csv"
    get_embeddings_with_hearbaselines(dataset_csv, audio_folder, output_folder)
    
    
    print('stop')
    # ge.get_embeddings(master_csv,outputs_folder)

## 2 - Aggregate embeddings over time:
# averaged_time_embeddings_folder = os.path.join("/homes/in304/rank-based-embeddings/RankBasedLoss_for_DCASEasc2016_dataset","averaged_time_VGGish_embeddings")

# if not os.path.exists(averaged_time_embeddings_folder):
#     os.makedirs(averaged_time_embeddings_folder)


# manipulate_embeddings.compute_embedding_vectors_per_file(master_csv, outputs_folder, averaged_time_embeddings_folder, aggr_mode="mean")
# print('Finished computing averaged vector embeddings')

# # ## 3 - Compute distances:
# # examples_per_class_name = df.select_examples_per_attribute(master_csv, attribute='individual_id')

# # ## 3.1 - Comppute within class distances:
# # within_class_average_distance_from_averaged_embeddings = manipulate_embeddings.compute_within_class_distance(corrected_csv, averaged_time_embeddings_folder, examples_per_class_name )  
# # within_class_average_distance_from_maxpooled_embeddings = manipulate_embeddings.compute_within_class_distance(corrected_csv, maxpooled_embeddings_folder, examples_per_class_name ) 
# #     # Save these!
# # with open(os.path.join(outputs_folder,"within_class_average_distance_from_averaged_embeddings_dict"), 'w') as fp:
# #     json.dump(within_class_average_distance_from_averaged_embeddings, fp)
# # with open(os.path.join(outputs_folder, "within_class_average_distance_from_maxpooled_embeddings_dict"), 'w') as fp:
# #     json.dump(within_class_average_distance_from_maxpooled_embeddings, fp)

# # ## 3.2 - Compute inter-class distances:
# # inter_class_average_distance_from_averaged_embeddings = manipulate_embeddings.compute_inter_class_distance(corrected_csv, averaged_time_embeddings_folder, examples_per_class_name )
# # inter_class_average_distance_from_maxpooled_embeddings = manipulate_embeddings.compute_inter_class_distance(corrected_csv, maxpooled_embeddings_folder, examples_per_class_name )
# #     # Save these!
# # with open(os.path.join(outputs_folder,"inter_class_average_distance_from_averaged_embeddings_dict"), 'w') as fp:
# #     json.dump(inter_class_average_distance_from_averaged_embeddings, fp)
# # with open(os.path.join(outputs_folder,"inter_class_average_distance_from_maxpooled_embeddings_dict"), 'w') as fp:
# #     json.dump(inter_class_average_distance_from_maxpooled_embeddings, fp)





# # # ## 4 - plot distance matrix:





# # classes_dict = json.load(open(os.path.join(outputs_folder, "IDclasses_per_species.json")))
# # within_class_average_distance_from_averaged_embeddings = json.load(open(os.path.join(outputs_folder, "within_class_average_distance_from_averaged_embeddings_dict")))
# # inter_class_average_distance_from_averaged_embeddings = json.load(open(os.path.join(outputs_folder,"inter_class_average_distance_from_averaged_embeddings_dict")))


# # dist_matrix_averaged_embeddings, list_classes = manipulate_embeddings.make_distances_matrix_from_inter_and_intra_classes_disctances_dict(within_class_average_distance_from_averaged_embeddings, inter_class_average_distance_from_averaged_embeddings, classes_dict)
# # # dist_matrix_maxpooled_embeddings, list_classes = manipulate_embeddings.make_distances_matrix_from_inter_and_intra_classes_disctances_dict(within_class_average_distance_from_maxpooled_embeddings, inter_class_average_distance_from_maxpooled_embeddings, classes_dict)


# # np.save( os.path.join(outputs_folder, 'distance_matrix.npy'), dist_matrix_averaged_embeddings)

# # dist_matrix_averaged_embeddings = np.load(os.path.join(outputs_folder, 'distance_matrix.npy'))




# # show_results.plot_heatmap_distance_matrix(dist_matrix_averaged_embeddings, list_classes, os.path.join(outputs_folder, 'dist_matrix_averaged_embeddings.png'))
# # # show_results.plot_heatmap_matrix(dist_matrix_maxpooled_embeddings, list_classes, os.path.join(outputs_folder,'dist_matrix_maxpooled_embeddings.png'))


