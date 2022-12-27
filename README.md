# I Hear Your True Colors: Image Guided Audio Generation
This repo contains the official PyTorch implementation of the pipeline presented in *I Hear Your True Colors: Image Guided Audio Generation*: [Paper](https://arxiv.org/abs/2211.03089), [Project page](https://pages.cs.huji.ac.il/adiyoss-lab/im2wav/).

# Abstract
We propose Im2Wav, an image guided open-domain audio generation system. Given an input image or a sequence of images, Im2Wav generates a semantically relevant sound. Im2Wav is based on two Transformer language models, that operate over a hierarchical discrete audio representation obtained from a VQ-VAE based model. We first produce a low-level audio representation using a language model. Then, we upsample the audio tokens using an additional language model to generate a high-fidelity audio sample. We use the rich semantics of a pre-trained CLIP embedding as a visual representation to condition the language model. In addition, to steer the generation process towards the conditioning image, we apply the classifier-free guidance method. Results suggest that Im2Wav significantly outperforms the evaluated baselines in both fidelity and relevance evaluation metrics. Additionally, we provide an ablation study to better assess the impact of each of the method components on overall performance. Lastly, to better evaluate image-to-audio models, we propose an out-of-domain image dataset, denoted as ImageHear. ImageHear can be used as a benchmark for evaluating future image-to-audio models.
<br> <br> <br>
![Pipeline overview](etc/Architechture.svg)
<br>

# Installation
```
git clone git@github.com:RoySheffer/im2wav.git
cd im2wav
pip install -r requirements.txt
```
Note: torch installation may depend on your cuda version. see [Install torch](https://pytorch.org/get-started/locally/)

# Usage
We provide a toy example using two videos and two single images. The same scripts can be used for the full VGGSound or any other custom dataset.
We additionally include the ImageHear dataset under `Data/ImageHear/` folder.

We start by setting the directory where all scripts should be run from:<br>
```
mkdir run && cd run
```

Next, we collect the CLIP image representations:<br>

### Collect CLIP representations of video directory
```
python ../Data/preprocess/collect_video_CLIP.py \
-videos_dir ../Data/examples/video
```

### Collect CLIP representations of images
```
python ../Data/preprocess/collect_image_CLIP.py \
-path_list ../Data/ImageHear/bongo1.jpg  ../Data/ImageHear/dog1.jpg
```

## Train
Train the models:<br>
* Set a batch size (bs) according to your GPU size.
### Train VQ-VAE
```
python ../models/train.py \
--hps=small_multi_level_vqvae \
--name=im2wav_vq \
--sample_length=65536 \
--bs=2 \
--audio_files_dir=../Data/examples/wav \
--labels=False \
--train \
--aug_shift \
--aug_blend \
```

### Train Low model
```
python ../models/train.py \
--hps=small_multi_level_vqvae,small_labelled_prior,all_fp16,cpu_ema \
--name=im2wav_low \
--sample_length=65536 \
--n_ctx=2048 \
--bs=2 \
--aug_shift \
--aug_blend \
--audio_files_dir=../Data/examples/wav \
--labels=True \
--train \
--test \
--prior \
--restore_vqvae=logs/im2wav_vq/checkpoint_latest.pth.tar \
--levels=2 \
--level=1 \
--weight_decay=0.01 \
--save_iters=2 \
--file2CLIP=video_CLIP \
--clip_emb  \
--video_clip_emb \
--class_free_guidance_prob=0.5
```

### Train Up model
```
python ../models/train.py \
--hps=small_multi_level_vqvae,small_upsampler,all_fp16,cpu_ema \
--name=im2wav_up \
--sample_length=65536 \
--n_ctx=8192 \
--bs=2 \
--audio_files_dir=../Data/examples/wav \
--labels=True \
--train \
--test \
--aug_shift \
--aug_blend \
--save_iters=2 \
--prior \
--restore_vqvae=logs/im2wav_vq/checkpoint_latest.pth.tar \
--file2CLIP=video_CLIP \
--levels=2 \
--level=0 \
--clip_emb
```

## Sample
After the models converge, we can use the trained models for an audio generation as follows:<br>

### Video condition sampling
```
python ../models/sample.py \
-bs 2 \
-experiment_name video_CLIP \
-CLIP_dir video_CLIP \
-models my_model
```

### Image condition sampling
```
python ../models/sample.py \
-bs 2 \
-wav_per_object 2 \
-experiment_name image_CLIP \
-CLIP_dict image_CLIP/CLIP.pickle \
-models my_model
```

## Use pre-trained model
We start by setting the directory where the pre-trained model weights should be downloaded to:
```
mkdir ../pre_trained
```

### Download the pre-trained model weights
```
pip install gdown

gdown 1lCrGsMXqmeKBk-3B3J2jzxNur9olWseb -O ../pre_trained/
gdown 1v9dmCwrEwkwJhbe2YF3ScM2gjVplSLzt -O ../pre_trained/
gdown 1UyNBjoxgqBYqA_aYhOu6BHYlkT4CD_M_ -O ../pre_trained/
```

### Sampling from pre-trained model
Repeat the Video/Image condition sampling steps replacing my_model with im2wav.

# Cite
If you find this implementation useful please consider citing our work:
```
@misc{sheffer2022i,
    title={I Hear Your True Colors: Image Guided Audio Generation},
    author={Roy Sheffer and Yossi Adi},
    year={2022},
    eprint={2211.03089},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```

## License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file. Some of the code in models dir was adapted from the [JukeBox](https://github.com/openai/jukebox) repository. 

