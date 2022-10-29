import sys
import os
modules_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.append(modules_dir)
import skvideo.io
import torch
from PIL import Image
import clip
import numpy as np
import pickle
import glob
import traceback as tb
import math
from models.hparams import CLIP_VERSION
import argparse


def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


def read_video(video, global_info):
    try:
        video_name = os.path.basename(video).split('.')[0]
        videodata = skvideo.io.vread(video)
        videometadata = skvideo.io.ffprobe(video)
        frame_rate = videometadata['video']['@avg_frame_rate']
        frame_num = videodata.shape[0]
        frames_in_sec = convert(frame_rate)
        length_in_secs = frame_num / frames_in_sec

        if global_info["videos_length"] is not None:
            if length_in_secs != global_info["videos_length"]:
                print(f"{video} video length: {frame_num} frames {length_in_secs} secs filtered\n\n")
                # os.remove(video)
                return [None, None, None, video_name]
        return [videodata, length_in_secs, frame_num, video_name]

    except Exception as e:
        err_msg = '{} Error while reading video: {}; \n{} {}'.format(video, e, tb.format_exc(),"\n-----------------------------------------------------------------------\n")
        print(err_msg)
        # os.remove(video)
        return [None, None, None, None]


def get_video_clip(video, device, model, global_info):
    with torch.no_grad():
        images = torch.cat([global_info["preprocess"](frame).unsqueeze(0).to(device) for frame in video])
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
        return image_features

def handle_video(video, global_info):
    try:
        video_name = os.path.basename(video).split('.')[0]
        save_dir = global_info["save_dir"]
        pickle_name = f"{save_dir}/{video_name}.pickle"
        if os.path.exists(pickle_name):
            return
        videodata, length_in_secs, frame_num, video_name = read_video(video, global_info)
        video_to_embed = [Image.fromarray(frame) for frame in videodata]
        image_features = get_video_clip(video_to_embed, global_info["device"], global_info["model"], global_info)
        file2CLIP = {}
        file2CLIP[video_name] = image_features
        with open(pickle_name, 'wb') as handle:
            pickle.dump(file2CLIP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        err_msg = 'Error while processing video {}: {}; {}'.format(video, e, tb.format_exc())
        print(err_msg)

def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, str])
    """
    parser = argparse.ArgumentParser(description='collect CLIP')
    parser.add_argument("-save_dir", dest='save_dir', action='store', type=str, default="video_CLIP")
    parser.add_argument("-videos_dir", dest='videos_dir', action='store', type=str)
    parser.add_argument("-videos_length", dest='videos_length', action='store', type=float)
    parser.add_argument("-bs", dest='bs', action='store', type=float, default=10)
    parser.add_argument("-multi_thread", dest='multi_thread', action='store', type=bool)

    v = vars(parser.parse_args())
    print(v)
    return v

if __name__ == '__main__':
    print("running CLIP collection")
    global_info = parse_arguments()

    if global_info["multi_thread"] is not None:
        import multiprocessing as mp
        import atexit
        atexit.register(lambda: os.system('stty sane') if sys.stdin.isatty() else None)

    global_info["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(global_info["save_dir"], exist_ok=True)
    global_info["videos"] = glob.glob(os.path.join(global_info["videos_dir"], '*.mp4'))
    global_info["videos_num"] = len(global_info["videos"])

    global_info["num_batches"] = math.ceil(global_info["videos_num"] / float(global_info["bs"]))
    print("assert ", (global_info["num_batches"]-1) * global_info["bs"], "<", global_info["videos_num"], "<", global_info["num_batches"] * global_info["bs"])
    print("using " , global_info["videos_num"], " videos")

    sys.stdout.flush()
    with torch.no_grad():
        global_info["model"], global_info["preprocess"] = clip.load(CLIP_VERSION, device=global_info["device"])
        if global_info["multi_thread"] is not None:
            max_num_workers = mp.cpu_count()
            num_workers = min(max_num_workers,10)
            print(f"using {num_workers} workers max_num_workers={max_num_workers}")
            for b in range(global_info["num_batches"]):
                num_batches = global_info["num_batches"]
                print(f"batch {b+1} out of {num_batches} ")
                batch_videos = global_info["videos"][b*global_info["bs"]:(b+1)*global_info["bs"]]
                with mp.Pool(num_workers) as pool:
                    try:
                        for i, video in enumerate(batch_videos):
                            pool.apply_async(handle_video, args=(video, global_info))
                    except Exception as e:
                        err_msg = 'Encountered error in {} at line: {}'
                        sys.exit(err_msg.format(video, e))
                    finally:
                        pool.close()
                        pool.join()
                sys.stdout.flush()
        else:
            for i, video in enumerate(global_info["videos"]):
                handle_video(video, global_info)
                if i % 1000 == 0:
                    print(f"finished {i+1}")
                    sys.stdout.flush()
    print("finished")





