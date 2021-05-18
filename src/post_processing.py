import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_curve
from scipy.spatial.distance import directed_hausdorff

from skimage.measure import inertia_tensor_eigvals, inertia_tensor, perimeter
from data_generator import DataGenerator


### Post-processing ###
def to_binary_results(arr: np.array) -> np.array:
    """Transforms the multiclass array into a binary classification
    task : class 1 vs. background"""

    # sum of two regions composed of non-overlapping pixels of value 1
    unified_class = arr[:, :, :, 1] + arr[:, :, :, 2]

    binary_mat = np.zeros((*arr.shape[:-1], 2))
    binary_mat[:, :, :, 0] = arr[:, :, :, 0]  # background
    binary_mat[:, :, :, 1] = unified_class

    return binary_mat


def get_predictions_labels(image_data_path: str,
                           mask_data_path: str,
                           k_fold_partition: list,
                           predictions: np.array,
                           multilabel: bool = False) -> (np.array, np.array):
    """Get le labels/maks for each sample.
    For sample sand labels, returns an binary array strucured as [nb_samples, height, width, nb_class]
    or an array [nb_samples, height, width] with nb_class unique values 
    if the MULITLABEL parameter is TRUE"""

    dim = (128, 128, 1)
    # create a generator in testing mode
    gen = DataGenerator(k_fold_partition, image_data_path,
                        mask_data_path, dim=dim,
                        testing=True)
    y_pred_raw = predictions
    y_true_raw = gen.get_labels()  # get the labels

    # if the labels should be encoded in a single plane (with nb_class different
    # values)
    if multilabel:
        y_true = y_true_raw.argmax(axis=3).flatten()
        y_pred = y_pred_raw.argmax(axis=3).flatten()
        return y_pred_raw, y_true_raw, y_pred, y_true

    else:
        return y_pred_raw, y_true_raw


def gather_predictions(image_data_path: list,
                       mask_data_path: list,
                       predictions_k_folds: list,
                       k_fold_partition: list) -> (np.array, np.array):
    """Make a single array gathering of all k-folds predictions and list of ids"""

    for k in range(5):
        if k > 0:
            # get the prediction realted to the samples
            preds_tmp, labels_tmp = get_predictions_labels(image_data_path,
                                                           mask_data_path,
                                                           k_fold_partition[k],
                                                           predictions_k_folds[k])
            # append the prediction to the other folds
            preds_ks = np.vstack((preds_ks, preds_tmp))
            labels = np.vstack((labels, labels_tmp))
        else:  # first ietration, create the array
            preds_ks, labels = get_predictions_labels(image_data_path,
                                                      mask_data_path,
                                                      k_fold_partition[k],
                                                      predictions_k_folds[k])

    return preds_ks, labels


class Metrics:
    """
    This class manages the evaluation of the model. Several metrics are used for each class : 
    - recall
    - precision
    - f1 score
    - specificity
    - intersection over union (i_o_u)
    - hausdorff score
    - accuracy
    """

    def __init__(self, y_true_raw: np.array,
                 y_preds_raw: np.array):
        """Receives the raw predictions (non-discretized) and the labels"""

        # discretize the predictions into a single plane of n_class unique values
        self.y_true = y_true_raw.argmax(axis=3).flatten()
        self.y_pred = y_preds_raw.argmax(axis=3).flatten()

        self.y_true_raw = y_true_raw
        self.y_preds_raw = y_preds_raw

        # overall
        cnf_matrix = confusion_matrix(self.y_true, self.y_pred)

        # comnputes the intermediary metrics
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  # false positive
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)  # false negative
        TP = np.diag(cnf_matrix)  # true positive
        TN = cnf_matrix.sum() - (FP + FN + TP)  # true negative

        # to float for further operations
        self.FP = FP.astype(float)
        self.FN = FN.astype(float)
        self.TP = TP.astype(float)
        self.TN = TN.astype(float)

    def recall(self) -> float:
        "Recall"
        # Sensitivity, hit rate, recall, or true positive rate
        return self.TP / (self.TP + self.FN)

    def specificity(self) -> float:
        "specificity"
        # Specificity or true negative rate
        return self.TN / (self.TN + self.FP)

    def precision(self) -> float:
        "Precision"
        # Precision or positive predictive value
        return self.TP / (self.TP + self.FP)

    def accuracy(self) -> float:
        "Overall accuracy"
        return (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)

    def f1(self) -> float:
        "F1 score or Dice score"
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def i_o_u(self) -> float:
        "Intersection over Union / Jaccard index"
        return self.TP / (self.TP + self.FP + self.FN)

    def HD_score(self, binary=False) -> [float]:
        "directed hausdorff socre for muilti or binary classification"
        truth = self.y_true_raw.argmax(axis=3)
        pred = self.y_preds_raw.argmax(axis=3)

        # background
        truth_0 = truth < 1
        pred_0 = pred < 1
        HD_0 = np.mean([directed_hausdorff(truth_tmp, pred_tmp)[0] for truth_tmp, pred_tmp in zip(truth_0, pred_0)])

        # Class1
        truth_1 = truth == 1
        pred_1 = pred == 1
        HD_1 = np.mean([directed_hausdorff(truth_tmp, pred_tmp)[0] for truth_tmp, pred_tmp in zip(truth_1, pred_0)])

        if not binary:
            # Class2
            truth_2 = truth > 1
            pred_2 = truth > 1
            HD_2 = np.mean([directed_hausdorff(truth_tmp, pred_tmp)[0] for truth_tmp, pred_tmp in zip(truth_2, pred_0)])

            return [HD_0, HD_1, HD_2]

        else:
            return [HD_0, HD_1]

    def confusion_matrix(self) -> pd.DataFrame:
        """Return the confusion matrix in a pd.DataFrame with two scales :
        - globally relative : divided by the overall number of samples
        - class relative : normalized by the number of true conditions for each class"""

        conf_mat = confusion_matrix(self.y_true, self.y_pred)
        conf_mat_percent = np.round(conf_mat * 100 / sum(sum(conf_mat)), 2)
        # the metrics normalized by the support of each true label (row-wise)
        conf_mat_relative = np.round(100 * (conf_mat.T / np.sum(conf_mat, axis=1)).T, 2)

        # if binary classification task
        if len(np.unique(self.y_true)) > 2:
            # create dataframes 
            conf_mat_percent = pd.DataFrame(data=conf_mat_percent, columns=['Background', 'Gray', 'White'],
                                            index=['Background', 'Gray', 'White']).rename_axis('True label')
            conf_mat_relative = pd.DataFrame(data=conf_mat_relative, columns=['Background', 'Gray', 'White'],
                                             index=['Background', 'Gray', 'White']).rename_axis('True label')
        else:
            # create dataframes 
            conf_mat_percent = pd.DataFrame(data=conf_mat_percent, columns=['Background', 'Spine'],
                                            index=['Background', 'Spine']).rename_axis('True label')
            conf_mat_relative = pd.DataFrame(data=conf_mat_relative, columns=['Background', 'Spine'],
                                             index=['Background', 'Spine']).rename_axis('True label')
        # get dataframes together
        mat = pd.concat((conf_mat_percent, conf_mat_relative),
                        axis=1, keys=['Global [%]', 'Relative [%]'])
        return mat

    def get_performance(self)-> pd.DataFrame:
        """Computes the metrics for each class and on average.
        Returns a pd.DataFrame containing all scores"""
        data = list(self.recall()) + list(self.precision()) + list(self.f1()) + \
               list(self.specificity()) + list(self.i_o_u()) + list(self.accuracy())

        if len(np.unique(self.y_true)) > 2:  # if multiclass classification
            # get performance for each class

            data = np.array(data + self.HD_score()).reshape((7, 3))

            # take the average of each metric
            avg = np.mean(data, axis=1).reshape(-1, 1)

            # average each metric by the weight of the class
            weighted_class = np.sum(np.sum(self.y_true_raw, axis=(1, 2)), axis=0)
            weighted_class = data.dot(weighted_class) / np.sum(weighted_class)

            # add to matrix
            data = np.hstack((data, avg))
            data = np.hstack((data, weighted_class.reshape(-1, 1)))

            # create nicer data format
            df_perf = pd.DataFrame(data=data,
                                   columns=['Background', 'Grey', 'White', 'Avg.', 'Weighted Avg.'],
                                   index=['Recall', 'Precision', 'F1 score', 'Specificity', 'IuO', 'Accuracy',
                                          'Hausdorff score'])
        else:
            # create nicer data format
            data = np.array(data + self.HD_score(binary=True)).reshape((7, 2))
            df_perf = pd.DataFrame(data=data,
                                   columns=['Background', 'Spine'],
                                   index=['Recall', 'Precision', 'F1 score', 'Specificity', 'IuO', 'Accuracy',
                                          'Hausdorff score'])

        return df_perf.round(3)


### Plotting ###

def plot_scores(scores: pd.DataFrame,
                err : pd.DataFrame,
                title: str) -> None:
    """Bar plot for each metric and each class from the
    Scores dataframe. Can add vertical error bars with matching ERR
    dataframe"""

    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    scores.drop('Hausdorff score', axis=0).plot.bar(ax=ax, yerr=err)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_ylim((0.70, 1.1))
    ax.set_title(title)
    fig.tight_layout()
    fig.show()


def plot_precision_recall_curve(y_true_raw: np.array,
                                y_pred_raw: np.array) -> None:
    """Pot the precision vs recall curve and the influence of
    threshold slection for binary classification"""
    plt.style.use('seaborn')
    fig = plt.figure(constrained_layout=False, figsize=(15, 8))

    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    axes = [ax1, ax2, ax3]

    n_classes = 3
    class_label = ['Background', "Grey Matter", 'White Matter']
    for i in range(n_classes):
        label_tmp = y_true_raw[:, :, :, i].flatten().astype(np.uint8)
        pred_tmp = y_pred_raw[:, :, :, i].flatten()
        precision_, recall_, thresholds_ = precision_recall_curve(label_tmp, pred_tmp)
        ax.plot(recall_, precision_, lw=2, linestyle='-', marker='.', label=class_label[i])

        axes[i].plot(thresholds_, precision_[:-1], lw=2, linestyle='-', marker='.', label='Precision')
        axes[i].plot(thresholds_, recall_[:-1], lw=2, linestyle='-', marker='.', label='Recall')
        axes[i].set_title(class_label[i])
        axes[i].legend()
        axes[i].set_xlabel('Thresholds')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision $vs$ Recall curve')
    ax.legend()

    axes[0].set_ylabel('Precision / Recall')

    fig.tight_layout()
    fig.show()


### Segmentation characteristics ###

def isolate_region(prediction: np.array, region: str):
    """Returns a binary image isolating a selected region"""
    if region == 'spine':
        binary_img = prediction > 0
    elif region == 'white':
        binary_img = prediction == 1
    elif region == 'grey':
        binary_img = prediction == 2
    else:
        raise ValueError("The region argument must be either 'spine, 'white' or 'grey'.")

    return binary_img.astype(np.uint8)


def get_area(prediction: np.array) -> np.array:
    """Returns the area, i.e. the number of pixels which value is 1"""
    return np.sum(prediction)


def get_height_width(prediction: np.array) -> (float, float):
    """Approximates the height and width of the white matter"""
    height, width = inertia_tensor_eigvals(prediction)
    return height, width


def get_rectangularity(height: float, width: float,
                       area: float) -> float:
    """Returns the rectangulariy"""
    if (height * width) == 0:
        raise ZeroDivisionError('Height or width is null')
    else:
        return area / (height * width)


def get_elongation(height: float, width: float) -> float:
    """Returns the elongation"""
    if width == 0:
        raise ZeroDivisionError('Width is null')
    else:
        return height / width


def get_compacity(perimeter_: float, area: float):
    """Returns the compacity"""
    if area ==0:
        raise ZeroDivisionError('Area is null')
    else:
        return (perimeter_ * perimeter_) / area


def get_orientation_angle(prediction: np.array) -> np.ndarray:
    """Returns the angle of orientation"""
    _, eigvecs = np.linalg.eigh(inertia_tensor(prediction))
    v = eigvecs[:, 0]
    angle = np.arctan(v[1] / v[0])
    return np.degrees(angle)


def get_perimeter(prediction: np.array) -> float:
    """Returns the perimeter"""
    return perimeter(prediction)
