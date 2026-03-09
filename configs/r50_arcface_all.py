from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0) # ArcFace loss
config.network = "r50"
config.resume = False
config.output = 'r50_arcface_all'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 1e-2
config.verbose = 2000
config.dali = False

config.img_list = 'ROOT_DIR/PetFace/split/all/train.csv'
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
