import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from utils.dataset import dataset
from utils.common import PSNR
from model import SRCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=200000, help='-')
parser.add_argument("--batch-size",     type=int, default=128,    help='-')
parser.add_argument("--architecture",   type=str, default="915",  help='-')
parser.add_argument("--save-every",     type=int, default=1000,   help='-')
parser.add_argument("--save-best-only", type=int, default=0,      help='-')
parser.add_argument("--save-log",       type=int, default=0,      help='-')
parser.add_argument("--ckpt-dir",       type=str, default="",     help='-')

FLAGS, unparsed = parser.parse_known_args()
steps = FLAGS.steps
batch_size = FLAGS.batch_size
save_every = FLAGS.save_every
save_log = (FLAGS.save_log == 1)
save_best_only = (FLAGS.save_best_only == 1)

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935 or 955")

ckpt_dir = FLAGS.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = f"checkpoint/SRCNN{architecture}"
model_path = os.path.join(ckpt_dir, f"SRCNN-{architecture}.h5")



# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
lr_crop_size = 33
hr_crop_size = 21
if architecture == "935":
    hr_crop_size = 19
elif architecture == "955":
    hr_crop_size = 17

train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

def main():
    model = SRCNN(architecture)
    model.setup(optimizer=Adam(learning_rate=2e-5),
                loss=MeanSquaredError(),
                model_path=model_path,
                metric=PSNR)

    model.load_checkpoint(ckpt_dir)
    model.train(train_set, valid_set, steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, save_every=save_every,
                save_log=save_log, log_dir=ckpt_dir)


if __name__ == "__main__":
    main()
