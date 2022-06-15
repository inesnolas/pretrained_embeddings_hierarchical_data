
import soundfile as sf
import torch

import hearbaseline.wav2vec2 as wav2vec
import hearbaseline.torchopenl3 as openl3
import hearbaseline.vggish as vggish 

import h5py
import os


model_wav2vec = wav2vec.load_model()
model_openl3 = openl3.load_model()
model_vggish = vggish.load_model()
# list_wavs = os.listdir('/import/c4dm-datasets/animal_identification/AAII_paper_augmented_dataset/chiffchaff-fg/')

audio, sr = sf.read('/import/c4dm-datasets/animal_identification/AAII_paper_augmented_dataset/chiffchaff-fg/cutted_day1_PC1107_0223.wav')
audio_tensor = torch.from_numpy( audio).reshape(1,-1).type(dtype=torch.FloatTensor)

emb_wav2vec2 = wav2vec.get_scene_embeddings(audio_tensor.cuda(), model_wav2vec) #embedding shape = 1024
emb_openl3 = openl3.get_scene_embeddings(audio_tensor.cuda(), model_openl3) #embeddings shpae = 6144
emb_vggish = vggish.get_scene_embeddings(audio_tensor.cuda(), model_vggish) #embedding shape = 128





print('hello')