import sys
import os
modules_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.append(modules_dir)
from im2wav_utils import *
from models.utils.dist_utils import setup_dist_from_mpi
from models.utils.torch_utils import empty_cache
import math
import time
import argparse
from models.trained_models import *
from models.hparams import SAMPLE_RATE, VIDEO_TOTAL_LENGTHS, VIDEO_FPS

# Break total_length into hops/windows of size n_ctx separated by hop_length
def get_starts(total_length, n_ctx, hop_length):
    starts = []
    for start in range(0, total_length - n_ctx + hop_length, hop_length):
        if start + n_ctx >= total_length:
            # Last hop could be smaller, we make it n_ctx to maximise context
            start = total_length - n_ctx
        starts.append(start)
    return starts


def adjust_y(y, prior, start, total_sample_length):
    if y is None:
        print(f"y is None")
        return y
    if len(y.shape) == 3: # (batch, frames, y ) y[2] = [total_length, offset, sample_length, clip]
        y[:, :, 2] = int(prior.sample_length)  # Set sample_length to match this level
        offset = int(start * prior.raw_to_tokens)
        y[:, :, 1:2] = offset
        offset_in_sec, length_in_sec = offset / float(SAMPLE_RATE), prior.sample_length / float(SAMPLE_RATE)
        print(offset_in_sec ,VIDEO_FPS)
        start = int(offset_in_sec * VIDEO_FPS)
        end = int(start + length_in_sec * VIDEO_FPS)
        print(f"using frames [{start}, {end}] out of the total {y.shape[1]} frames {(end - start)/y.shape[1]} =? {prior.sample_length / total_sample_length}")
        y = y[:, start:end + 1]
    elif len(y.shape) == 2: # (batch, y) y[1] = [total_length, offset, sample_length, clip]
        y[:, 2] = int(prior.sample_length)  # Set sample_length to match this level
        offset = int(start * prior.raw_to_tokens)  # Set offset
        y[:, 1:2] = offset
    return y


def multi_level_sample_window(priors, y, y_video, n_samples, top_k, top_p, cfg_s, hop_fraction, total_sample_length, sliding_window=False):
    sample_levels = list(range(len(priors)))
    zs = [torch.zeros(n_samples, 0, dtype=torch.long, device=device) for _ in range(len(priors))]
    xs = []
    for level in reversed(sample_levels):
        prior = priors[level]
        if prior is None:
            continue
        if torch.cuda.is_available():
            prior.cuda()

        if prior.video_clip_emb:
            y_hat = y_video
        else:
            y_hat = y

        empty_cache()

        assert total_sample_length % prior.raw_to_tokens == 0, f"Expected sample_length {total_sample_length} to be multiple of {prior.raw_to_tokens}"

        if sliding_window:
            # Set correct total_length, hop_length, labels and sampling_kwargs for level
            total_length = total_sample_length//prior.raw_to_tokens
            hop_length = int(hop_fraction[level]*prior.n_ctx)

            for start in get_starts(total_length, prior.n_ctx, hop_length):
                end = start + prior.n_ctx
                z = zs[level][:, start:end]

                sample_tokens = (end - start)
                conditioning_tokens, new_tokens = z.shape[1], sample_tokens - z.shape[1]
                print(f"Sampling {sample_tokens} tokens for [{start},{start + sample_tokens}]. Conditioning on {conditioning_tokens} tokens z.shape={z.shape}")
                if new_tokens <= 0:
                    # Nothing new to sample
                    continue
                # get z_conds from level above
                z_conds = prior.get_z_conds(zs, start, end)

                # set y offset, sample_length and lyrics tokens
                y_cur = adjust_y(y_hat.clone(), prior, start, hps['total_sample_length'])
                empty_cache()

                z = prior.sample(n_samples=n_samples, z=z, z_conds=z_conds, y=y_cur, top_k=top_k, top_p=top_p, cfg_s=cfg_s)
                # Update z with new sample
                z_new = z[:, -new_tokens:]
                zs[level] = torch.cat([zs[level], z_new], dim=1)
        else:
            start = 0
            end = start + prior.n_ctx
            z = zs[level][:, start:end]
            # get z_conds from level above
            z_conds = prior.get_z_conds(zs, start, end)
            y_cur = adjust_y(y_hat, prior, start, hps['total_sample_length'])
            zs[level] = prior.sample(n_samples=n_samples, z=z, z_conds=z_conds, y=y_cur, top_k=top_k, top_p=top_p, cfg_s=cfg_s)

        prior.cpu()
        empty_cache()

        x = prior.decode(zs[level:], start_level=level, bs_chunks=zs[level].shape[0])  # Decode sample
        xs.append(x)
    return xs


def get_y(video, pos ,clip_emb):
    if video:
        pose_tiled = np.tile(pos, (clip_emb.shape[0], 1))
        y = np.concatenate([pose_tiled, clip_emb], axis=1, dtype=np.float32)
    else:
        y = np.concatenate([pos, clip_emb.flatten()], dtype=np.float32)
    return y


def save_samples(hps, sliding_window=False):
    resultsDir = os.path.join(hps['save_dir'], hps["experiment_name"], hps["resultsDir"], hps["model_name"])
    first = 0
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
        for level in [0,1]:
            os.makedirs(f"{resultsDir}/l{level}")

    if hps["first"] is not None:
        if first < hps["first"]:
            first = hps["first"]

    with torch.no_grad():
        print(f"using {device} hps:{hps}")

        TRIES = 10
        for tr in range(TRIES):
            try:
                vqvae = get_model_from_checkpoint(hps["vq_cp"], device)
                prior = get_model_from_checkpoint_prior(hps["prior_cp"], vqvae, device)
                upsampler = get_model_from_checkpoint_prior(hps["up_cp"], vqvae, device)
                break
            except Exception as e:
                print(tr, ": ------------------ \n", e)
                time.sleep(5)

        priors = [upsampler, prior]

        print(hps["model_name"], f"starting from {first} top_k:", hps["top_k"], "top_p:", hps["top_p"] ,"  --------------------------------------------------------------------")
        for i in range(hps["num_batches"]):
            if i * hps["bs"] < first:
                continue

            start, end = i * hps["bs"], (i + 1) * hps["bs"]
            end = min(end, ys.shape[0])
            required = 0
            for m in range(start, end):
                if not os.path.exists(f"{resultsDir}/l0/{file_names[m]}.wav") or not os.path.exists(f"{resultsDir}/l1/{file_names[m]}.wav"):
                    required += 1
            if required == 0:
                print(f"skipping [{start}, {end}]")
                continue
            else:
                print(f"required {required} in [{start}, {end}]")

            y = ys[start:end]
            y_video = ys_video[start:end]

            y = y.to(device, non_blocking=True)
            y_video = y_video.to(device, non_blocking=True)

            xs = multi_level_sample_window(priors, y, y_video, y.shape[0], hps["top_k"], cfg_s=hps["cfg_s"], top_p=hps["top_p"], hop_fraction=hps["hop_fraction"], total_sample_length=hps['total_sample_length'], sliding_window=sliding_window)

            for level, x_sample in enumerate(xs):
                for j in range(y.shape[0]):
                    index = i * hps["bs"] + j
                    if index < first:
                        continue
                    audio = x_sample[j, :, 0].cpu().numpy()
                    name = f"{resultsDir}/l{level}/{file_names[index]}.wav"
                    soundfile.write(name,audio, samplerate=SAMPLE_RATE, format='wav')


def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, str])
    """
    parser = argparse.ArgumentParser(description='sample')

    parser.add_argument("-experiment_name", dest='experiment_name', action='store', type=str, default="no_name")
    parser.add_argument("-save_dir", dest='save_dir', action='store', type=str, default="samples")
    parser.add_argument("-model_name", dest='model_name', action='store', type=str, default="my_model")

    parser.add_argument("-vq_cp", dest='vq_cp', action='store', type=str)
    parser.add_argument("-prior_cp", dest='prior_cp', action='store', type=str)
    parser.add_argument("-up_cp", dest='up_cp', action='store', type=str)

    parser.add_argument("-resultsDir", dest='resultsDir', action='store', type=str, default="")
    parser.add_argument("-bs", dest='bs', action='store', type=int, default=4)

    parser.add_argument("-first", dest='first', action='store', type=int)
    parser.add_argument("-cfg_s", dest='cfg_s', action='store', type=float, default=3.0)

    parser.add_argument("-wav_per_object", dest='wav_per_object', action='store', type=int, default=120)

    parser.add_argument("-p_grid", dest='p_grid', action='store', type=float, nargs="+", default=[0])
    parser.add_argument("-k_grid", dest='k_grid', action='store', type=int, nargs="+", default=[0])

    parser.add_argument("-CLIP_dir", dest='CLIP_dir', action='store', type=str)
    parser.add_argument("-CLIP_dict", dest='CLIP_dict', action='store', type=str)

    parser.add_argument("-models", dest='models', action='store', type=str, nargs="+", default=[])

    parser.add_argument("-hop_fraction", dest='hop_fraction', action='store', type=float, nargs="+", default=None)
    parser.add_argument("-sample_length", dest='sample_length', action='store', type=int, default=65536)
    parser.add_argument("-sliding_window_total_sample_length", dest='sliding_window_total_sample_length', action='store', type=int)

    v = vars(parser.parse_args())
    print(v)
    assert (v["CLIP_dict"] is None) or (v["CLIP_dir"] is None)
    return v


if __name__ == '__main__':
    hps = parse_arguments()
    rank, local_rank, device = setup_dist_from_mpi(port=(29500 + np.random.randint(99)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [name2model[model_name] for model_name in hps["models"]]
    if len(models):
        hps['sample_length'] = models[0]["sample_length"]

    if hps['sliding_window_total_sample_length'] is None:  #single generation not sliding window
        sliding_window = False
        hps['total_sample_length'] = hps['sample_length']
    else:
        sliding_window = True
        hps['total_sample_length'] = hps['sliding_window_total_sample_length']


    duration_in_sec = hps['total_sample_length']/SAMPLE_RATE
    offset = 0.0
    pos = np.array([VIDEO_TOTAL_LENGTHS, offset, hps['sample_length']], dtype=np.int64) # all current generation models should share the same sample_length

    if hps["CLIP_dir"]:
        ys, ys_video, video_names = [], [], []
        CLIP_paths = glob.glob(os.path.join(hps["CLIP_dir"], '*.pickle'))
        CLIP_paths = sorted(CLIP_paths) # to be able to rerun killed jobs based on indexes
        for CLIP_path in CLIP_paths:
            video_name = os.path.basename(CLIP_path).split('.')[0]
            if not os.path.exists(CLIP_path):
                continue
            with open(CLIP_path, 'rb') as handle:
                clip_emb = pickle.load(handle)[video_name]
            clip_emb= clip_emb[:int(duration_in_sec*VIDEO_FPS)]
            mean_sample_clip_emb = np.mean(clip_emb, axis=0)
            pose_tiled = np.tile(pos, (clip_emb.shape[0], 1))
            y_video = np.concatenate([pose_tiled, clip_emb], axis=1, dtype=np.float32)
            y = np.concatenate([pos, mean_sample_clip_emb.flatten()], dtype=np.float32)
            ys.append(y)
            ys_video.append(y_video)
            video_names.append(video_name)
        ys = torch.from_numpy(np.stack(ys))
        ys_video = torch.from_numpy(np.stack(ys_video))
        file_names = video_names
    else:
        with open(hps["CLIP_dict"], 'rb') as handle:
            CLIP_dict = pickle.load(handle)
            CLIP = CLIP_dict["image"]
        objects = list(CLIP.keys())
        chosen_objects = {"objects": [], "indexs": [], "clip_emb": [], "total_length": VIDEO_TOTAL_LENGTHS}
        clip_emb_all = []
        class_list = CLIP.keys()
        class_list = list(class_list)
        class_list.sort()
        for class_name in class_list:
            chosen_objects["objects"] += ([class_name] * hps["wav_per_object"])
            class_images_num = CLIP[class_name].shape[0]
            class_indices = list(range(class_images_num)) * int(np.ceil(hps["wav_per_object"] / class_images_num))
            class_indices = class_indices[: hps["wav_per_object"]]
            class_indices.sort()
            class_clip_emb = [CLIP[class_name][class_indices[i]] for i in range(len(class_indices))]
            chosen_objects["indexs"] += class_indices
            clip_emb_all += class_clip_emb
        clip_emb_all = np.array(clip_emb_all)
        ys = [get_y(video=False, pos=pos, clip_emb=clip_emb_all[i]) for i in range(clip_emb_all.shape[0])]
        ys = torch.from_numpy(np.stack(ys))
        ys_video = ys.reshape((ys.shape[0], 1, ys.shape[1]))
        file_names = []
        for i in range(ys.shape[0]):
            class_cur = chosen_objects["objects"][i]
            index_cur = chosen_objects["indexs"][i]
            name = f"{i}_{class_cur}_{index_cur}"
            file_names.append(name)
        hps["num_batches"] = math.ceil(len(chosen_objects["objects"]) / float(hps["bs"]))
    hps["num_batches"] = math.ceil(ys.shape[0] / float(hps["bs"]))

    if len(models) !=0:
        for k in hps["k_grid"]:
            for p in hps["p_grid"]:
                hps["top_k"] = k
                hps["top_p"] = p
                hps["resultsDir"] = ""
                if len(hps["k_grid"]) + len( hps["p_grid"]):
                    hps["resultsDir"]+=f"k_top{k}_p_top{p}"
                for model in models:
                    hps_cur = hps.copy()
                    for key in model:
                        hps_cur[key] = model[key]
                    save_samples(hps_cur, sliding_window=sliding_window)
    else:
        save_samples(hps, sliding_window=sliding_window)
    print("finished")