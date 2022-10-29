trained_models = []

my_model = {"vq_cp": "logs/im2wav_vq/checkpoint_latest.pth.tar",
       "prior_cp": "logs/im2wav_low/checkpoint_latest.pth.tar",
       "up_cp": "logs/im2wav_up/checkpoint_latest.pth.tar",
       "model_name": "my_model",
       "sample_length": 65536}
trained_models.append(my_model)

name2model = {}
for model in trained_models:
       name2model[model["model_name"]] = model
