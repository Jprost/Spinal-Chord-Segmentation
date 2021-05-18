import numpy as np
import tensorflow as tf
import nibabel as nib  # load data

from skimage.transform import resize
from scipy.stats import mode
from tqdm import tqdm


def to_numpy_file(data_path: str,
                  output_path: str,
                  files: str,
                  to_int: bool=False) -> None:
    """Loads a .nii.gz file and converts it to .npy"""
    for file in files:
        # loads the first mask and converts to int
        img = nib.load(data_path + file).get_fdata()
        if to_int:
            img = np.uint8(img)
        # save in npy format
        np.save(output_path + file.replace('nii.gz', 'npy'), img)


def crop_center(img: np.array, crop_shape: tuple) -> np.array:
    """Crop the image to a final arbitrary shape by removing borders equally"""
    cropx, cropy = crop_shape
    y, x = img.shape[0], img.shape[1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, ]


def to_slice_images(data_path: str,
                    output_path: str,
                    images_files: str,
                    shape: tuple) -> None:
    """Loads 3D images and split it into 2D images.
    Resize the images and save them with a -x.npy tag with indicates the slice
    number x."""
    for img_f in tqdm(images_files):
        img = np.load(data_path + img_f)
        for slice_ in range(img.shape[-1]):
            img_slice = img[:, :, slice_]

            if img_slice.shape[0] > shape[0]:
                img_slice = resize(img_slice, (3 * shape[0], 3 * shape[1]))
                img_slice = crop_center(img_slice, shape)
            else:
                img_slice = resize(img_slice, shape)

            if img_slice.max() > 0:
                np.save(output_path + img_f.replace('-image.npy', f'_{slice_}.npy'),
                        img_slice)


def create_2D_masks(data_path: str,
                    output_path: str,
                    mask_files: str,
                    shape: tuple) -> None:
    """Loads 3D masks to create a single mask for each slice.
    The rank of a final mask would be [height, widht, number of class]. In the
    last rank, the labels are binary encoded along an additional dimension
    per class.
    Masks are also resized to match the same transormation as their relative
    image.
    """
    for m_f in tqdm(mask_files):
        mask = np.load(data_path + m_f)
        for slice_ in range(mask.shape[-1]):
            mask_slice = mask[:, :, slice_]

            # if multiple labels are present on the mask, get rid miss-labeled
            if len(np.unique(mask_slice)) > 1:
                # from one hot encoding, to 3 binary planes, add a dimension
                mask_slice = tf.keras.utils.to_categorical(mask_slice, 3)

                # if the mask is larger than arbitrary shape
                if mask_slice.shape[0] > shape[0]:
                    # scale to 3 times the final shape
                    mask_slice = resize(mask_slice, (3 * shape[0], 3 * shape[1]))
                    # ensure to have binary pixels (resize implies interpolation)
                    mask_slice = (mask_slice > .5).astype(int)
                    # crop to final shape
                    mask_slice = crop_center(mask_slice, shape)
                else:  # if mask is smaller than the arbitrary shape
                    mask_slice = resize(mask_slice, shape)
                    mask_slice = (mask_slice > .5).astype(int)

                # save file
                np.save(output_path + m_f.replace('-mask.npy', f'_{slice_}.npy'),
                        mask_slice)


def mask_majority_voting(data_path: str,
                         output_path: str,
                         mask_files: str) -> None:
    """Loads the masks of the four raters and create a new mask based on
    majority voting for each voxel of each slice among the four provided
    masks.
    Simplifies teh name of the file to match the corresponding image name."""

    for mask_idx in tqdm(range(0, len(mask_files), 4)):
        # loads the first mask and converts to int
        mask = np.load(data_path + mask_files[mask_idx])
        masks_raw = np.zeros(((4,) + mask.shape))  # will load three other masks
        masks_raw[0,] = mask
        # concatenates the three other raters
        for rater in range(1, 4):
            masks_raw[rater,] = np.load(data_path + mask_files[mask_idx + rater])

        # for each slice, majority voting for the label 0, 1 or 2
        mask = np.zeros_like(mask)
        for slice_ in range(masks_raw.shape[-1]):
            raters_slice = masks_raw[:, :, :, slice_]
            mask[:, :, slice_] = mode(raters_slice, axis=0).mode

        np.save(output_path + mask_files[mask_idx][:15] + '.npy', mask)