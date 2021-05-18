import numpy as np
import tensorflow as tf

from scipy.ndimage import shift, rotate
from sklearn.model_selection import train_test_split


def _normalize_(x: np.array) -> np.array:
    """Normalize each frame along the height and width dimension"""
    if x.max() != 0:
        x = x / x.max()
        return np.clip(x, 0, 1)# ensure that no values are >1
    else:
        raise ZeroDivisionError('Image Normalization')


def _expand_dims_(x: np.array) -> np.array:
    """Expands the dimensions of the arrays along the last dimension"""
    x = np.expand_dims(x, axis=-1)
    return x


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data by batches for Keras models :
    - loads data
    - applies live random data augmentation
    Inherits from tf methods.
    Manages multiprocessing to load and transform data in parallel,
    ensuring that those steps are note the bottleneck factor that would
    limit computation speed.
    Called during the tf.model.fit()/predict() method."""

    def __init__(self, list_IDs: list,
                 image_data_path: str, mask_data_path: str,
                 dim: tuple,
                 shift: int=0, rotate: int=0,
                 batch_size: int=32, shuffle: bool=True,
                 testing: bool=False):

        """Initialization.
        Access to the data directories and set up the array
        processing transformations.
        Testing mode enables to have a simplified version of the generator
        without pre-processing."""

        self.list_IDs = list_IDs  # list of samples
        self.image_data_path = image_data_path  # path of images files
        self.mask_data_path = mask_data_path  # path of masked files
        self.indexes = np.empty(len(self.list_IDs))  # list of indexes initialized at each epoch

        self.testing = testing  # enable testing mode
        if testing:
            self.shuffle = False  # do not suffle IDs
            self.batch_size = 1  # batch size forced to 1for easier label access

        else:
            # pre processing is done only in non-testing mode
            self.shift = shift  # pixels to shift
            self.rotate = rotate  # angle in degree to rotate the image
            self.shuffle = shuffle  # shuffle IDs
            self.batch_size = batch_size

        self.dim = dim  # dimensions of each image
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    ### Pre processing ###
    def _preprocessing_(self, x: np.array, m: np.array) -> (np.array, np.array):
        """Applies transformation on the image and its corresponding mask"""
        #normalization
        x = _normalize_(x)

        if not self.testing:
            x, m = self._rotate_(x, m)
            x, m = self._width_shift_(x, m)
            x, m = self._height_shift_(x, m)

        x = _expand_dims_(x)
        return x, m

    def _width_shift_(self, x: np.array, m: np.array) -> (np.array, np.array):
        """
        Shift range. Use scipy method
        """
        # get a random sign for the shifting direction
        sign = np.random.randint(0, 2)
        shift_pix = np.random.randint(0, self.shift)
        x = shift(x, [0, sign*shift_pix])
        m = shift(m, [0, sign*shift_pix, 0], mode='nearest')
        return x,m

    def _height_shift_(self, x: np.array, m: np.array) -> (np.array, np.array):
        """
        Shift range. Use scipy method
        """
        # get a random sign for the shifting direction
        sign = np.random.randint(0, 2)
        shift_pix = np.random.randint(0, self.shift)
        x = shift(x, [0, sign*shift_pix])
        m = shift(m, [0, sign*shift_pix, 0], mode='nearest')
        return x, m

    def _rotate_(self, x: np.array, m: np.array) -> (np.array, np.array):
        """ Applies a random rotation of the image in the
        [ -'rotate_range' ; 'rotate_range' ] range to an image"""
        # get a random angle
        angle = np.random.randint(0, self.rotate)
        # get a random sign for the angle
        sign = np.random.randint(0, 2)
        x = rotate(x, -sign * angle, reshape=False)
        m = rotate(m, -sign * angle, axes=(0, 1),
                   mode='nearest',
                   reshape=False)
        return x, m

    ### Access data ###
    def __data_generation(self, list_IDs_temp: list) -> (np.array, np.array):
        """Generates data containing batch_size samples
         - X : (n_samples, *dim)
         - M : (n_samples, *dim, n_classes)"""

        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        # three channels fro three classes (one hot encoding)
        M = np.empty((self.batch_size, *self.dim[:-1], 3))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            x = np.load(self.image_data_path + ID, allow_pickle=True)
            m = np.load(self.mask_data_path + ID, allow_pickle=True)

            # pre processing
            x, m = self._preprocessing_(x, m)
            X[i,] = x
            M[i,] = m

        return X, M

    def __getitem__(self, index: list) -> (np.array, np.array):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, M = self.__data_generation(list_IDs_temp)

        return X, M

    def get_labels(self) -> np.array:
        """Returns the masks for each sample"""
        nb_samples = len(self.list_IDs)
        M = np.empty((nb_samples, *self.dim[:-1], 3))

        for i, ID in enumerate(self.list_IDs):
            M[i,] = np.load(self.mask_data_path + ID, allow_pickle=True)

        return M


def make_partition(X: list, Y: list,
                   val_frac: float,
                   random_state=None) ->(dict, dict):
    """Splits data into trainning and validation set"""

    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                      test_size=val_frac,
                                                      random_state=random_state)

    partition = {'train': X_train, 'val': X_val}
    labels = {'train': y_train, 'val': y_val}

    return partition, labels
