import numpy as np
from PIL import Image


def rand_crop(id, patch_size=572, num_patch=10):

    # todo: resize labels to (388, 388)
    image_path = 'inputs/train/' + id + '.jpg'
    mask_path = 'inputs/train_masks/' + id + '_mask.gif'
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    offset_w = np.random.randint(0, high=image.size[0] - patch_size, size=num_patch)
    offset_h = np.random.randint(0, high=image.size[1] - patch_size, size=num_patch)

    image_batch = np.zeros((num_patch, patch_size, patch_size, 3), dtype=np.uint8)
    label_batch = np.zeros((num_patch, patch_size, patch_size, 1), dtype=bool)

    idx = 0
    while idx < num_patch:
        left = offset_w[idx]
        right = offset_w[idx] + patch_size
        upper = offset_h[idx]
        lower = offset_h[idx] + patch_size

        mask_crop = mask.crop((left, upper, right, lower))

        if empty(mask_crop) is True:
            offset_w[idx] = np.random.randint(0, high=image.size[0] - patch_size)
            offset_h[idx] = np.random.randint(0, high=image.size[1] - patch_size)
        else:
            image_batch[idx, :, :, :] = image.crop((left, upper, right, lower))
            label_batch[idx, :, :, 0] = mask_crop
            idx += 1

    return image_batch, label_batch

def empty(mask, min_ratio = 0.2):
    num_ent = mask.size[0] * mask.size[0]
    num_nonzero = np.sum(mask)
    if num_nonzero / num_ent < min_ratio:
        return True
    else:
        return False

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    file_id = '0cdf5b5d0ce1_01'
    image_batch, label_batch = rand_crop(file_id, patch_size=572)

    index = np.random.randint(0, 10)

    plt.imshow(image_batch[index, :, :, :])
    plt.show()

    plt.imshow(label_batch[index, :, :, 0])
    plt.show()