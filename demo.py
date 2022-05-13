import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
from model import SRCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int, default=2,                   help='-')
parser.add_argument('--architecture', type=str, default="915",               help='-')
parser.add_argument("--ckpt-path",    type=str, default="",                  help='-')
parser.add_argument("--image-path",   type=str, default="dataset/test1.png", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935 or 955")

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.h5"

sigma = 0.3 if scale == 2 else 0.2
pad = int(architecture[1]) // 2 + 6


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = bicubic_image[pad:-pad, pad:-pad]
    write_image("bicubic.png", bicubic_image)

    lr_image = gaussian_blur(lr_image, sigma=sigma)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = rgb2ycbcr(bicubic_image)
    bicubic_image = norm01(bicubic_image)
    bicubic_image = tf.expand_dims(bicubic_image, axis=0)

    model = SRCNN(architecture)
    model.load_weights(ckpt_path)
    sr_image = model.predict(bicubic_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = tf.cast(sr_image, tf.uint8)
    sr_image = ycbcr2rgb(sr_image)
    write_image("sr.png", sr_image)

if __name__ == "__main__":
    main()
