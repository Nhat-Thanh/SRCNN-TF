from utils.common import *
from model import SRCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=float, default=2,                                  help='-')
parser.add_argument('--architecture', type=str,   default="915",                              help='-')
parser.add_argument("--image-path",   type=str,   default="dataset/test1.png",                help='-')
parser.add_argument("--ckpt-path",    type=str,   default="checkpoint/SRCNN915/SRCNN-915.h5", help='-')

FLAGS, unparsed = parser.parse_known_args()
architecture = FLAGS.architecture
image_path = FLAGS.image_path
ckpt_path = FLAGS.ckpt_path
scale = FLAGS.scale

if scale < 1 or scale > 5:
    ValueError("scale should be positive and less than 5")


# -----------------------------------------------------------
#  read image and save bicubic image
# -----------------------------------------------------------

lr_image = read_image(image_path)
bicubic_image = upscale(lr_image, scale)
write_image("bicubic.png", bicubic_image)


# -----------------------------------------------------------
# preprocess lr image 
# -----------------------------------------------------------

lr_image = gaussian_blur(lr_image, sigma=0.3)
lr_image = upscale(lr_image, scale)
lr_image = rgb2ycbcr(lr_image)
lr_image = norm01(lr_image)
lr_image = tf.expand_dims(lr_image, axis=0)


# -----------------------------------------------------------
#  predict and save image
# -----------------------------------------------------------

model = SRCNN(architecture)
model.load_weights(ckpt_path)
sr_image = model.predict(lr_image)[0]

sr_image = denorm01(sr_image)
sr_image = tf.cast(sr_image, tf.uint8)
sr_image = ycbcr2rgb(sr_image)

write_image("sr.png", sr_image)
