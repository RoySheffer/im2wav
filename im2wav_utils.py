import wav2clip
import numpy as np
import librosa
from torchvision import transforms
import torch
import clip
from PIL import Image
import pickle
from tqdm import tqdm
import soundfile
import math
from torch.nn import functional as F
from torch import nn, optim
import torchaudio.transforms as transforms
import torchvision.transforms as transformsVision
import sounddevice as sd
import torchaudio
import matplotlib.pyplot as plt
from models.make_models import make_vqvae, load_checkpoint, make_prior
from models.hparams import Hyperparams, setup_hparams, REMOTE_PREFIX, small_vqvae, DEFAULTS, small_prior
import glob
import os
import pandas as pd

def embed_audio(audio, model):
    if len(audio.shape) == 1:
        audio = audio.clone()[None, :]
    return (model(audio.to(next(model.parameters()).device)))

def get_audio_from_paths(sr, sample_length, paths):
    import models.utils.io as io
    audios, lengths = [], []
    for path in paths:
        try:
            audio, sr = io.load_audio(path, sr=sr, offset=0, duration=sample_length)
        except Exception as e:
            print(f"problem with {path}:\n {e} \n")
            continue
        audios.append(audio[0])
        lengths.append(audio[0].shape[0])
    lengths = np.array(lengths)
    audios = [audio[:np.min(lengths)] for audio in audios]
    return np.array(audios)


def get_model_from_checkpoint(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path)
    hps = checkpoint['hps']
    for key in checkpoint['hps']:
        hps[key] = checkpoint['hps'][key]
    hps['restore_vqvae'] = checkpoint_path
    hps = Hyperparams(hps)
    hps.train = False
    vqvae = make_vqvae(hps, device)
    return vqvae


def get_model_from_checkpoint_prior(checkpoint_path, vqae, device):
    if checkpoint_path == "":
        return None
    checkpoint = load_checkpoint(checkpoint_path)
    hps = checkpoint['hps']
    for key in checkpoint['hps']:
        hps[key] = checkpoint['hps'][key]
    hps['restore_prior'] = checkpoint_path
    hps = Hyperparams(hps)
    hps.train = False
    if "video_clip_emb" not in hps:
        hps.video_clip_emb = False
    if "class_free_guidance_prob" not in hps:
        hps.class_free_guidance_prob = -1
    prior = make_prior(hps, vqae, device)
    return prior


def encode_decode(vqvae, audio):
    y = vqvae._encode_noBottleneck(audio)
    z = vqvae.bottleneck.encode(y)
    X = vqvae._decode(z)
    # x_out, _, __ = vqvae(x_in, hps)
    print([m.shape for m in [audio, X]])
    print([m.shape for m in z])
    print([np.log2(audio.shape[1] / m.shape[1])  for m in z])
    return X


def parse_im2wav_name(name):
    attributes = name.split('_')
    if len(attributes)==3:
        index, class_name, class_image_index = attributes
    else:
        print(f"len({attributes})!=3")
        index, class_name, class_image_index = -1, "error", -1
    return index, class_name, class_image_index
