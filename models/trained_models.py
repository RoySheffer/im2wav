trained_models = []

my_model = {"vq_cp": "logs/im2wav_vq/checkpoint_latest.pth.tar",
       "prior_cp": "logs/im2wav_low/checkpoint_latest.pth.tar",
       "up_cp": "logs/im2wav_up/checkpoint_latest.pth.tar",
       "model_name": "my_model",
       "sample_length": 65536}
trained_models.append(my_model)

im2wav = {"vq_cp": "../pre_trained/vqvae.tar",
       "prior_cp": "../pre_trained/low.tar",
       "up_cp": "../pre_trained/up.tar",
       "model_name": "im2wav",
       "sample_length": 65536}
trained_models.append(im2wav)

name2model = {}
for model in trained_models:
       name2model[model["model_name"]] = model
