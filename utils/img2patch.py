import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rand_crop(id, patch_size=512, num_patch=10):
    image_path = 'inputs/train/' + id + '.jpg'
    mask_path = 'inputs/train_masks/' + id + '_mask.gif'
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    offset_w = np.random.randint(0, high=image.size[0] - patch_size, size=num_patch)
    offset_h = np.random.randint(0, high=image.size[1] - patch_size, size=num_patch)
    image_batch = np.zeros((num_patch, patch_size, patch_size, 3), dtype=np.uint8)
    label_batch = np.zeros((num_patch, patch_size, patch_size, 1), dtype=bool)

    for i in range(num_patch):
        left = offset_w[i]
        right = offset_w[i] + patch_size
        upper = offset_h[i]
        lower = offset_h[i] + patch_size

        image_batch[i, :, :, :] = image.crop((left, upper, right, lower))
        label_batch[i, :, :, 0] = mask.crop((left, upper, right, lower))
    return image_batch, label_batch

def drop_empty(mask, min_ratio = 0.2):
    
    pass

if __name__ == '__main__':
    file_id = '0cdf5b5d0ce1_01'
    # image = Image.open('inputs/29bb3ece3180_11.jpg')
    image_batch, label_batch = rand_crop(file_id, patch_size=512)

    index = np.random.randint(0, 10)

    plt.imshow(image_batch[index, :, :, :])
    plt.show()

    plt.imshow(label_batch[index, :, :, 0])
    plt.show()