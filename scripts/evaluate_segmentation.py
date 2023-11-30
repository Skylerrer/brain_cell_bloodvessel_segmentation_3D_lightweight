import numpy as np
from trpseg.trpseg_util.utility import get_file_list_from_directory, read_image


#Paper: "Towards a guideline for evaluation metrics in medical image segmentation": "Provide next to DSC also IoU, Sensitivity, and Specificity for method comparability."
#-> DSC, IoU, Sensitivity (= Recall), Specificity
#We additionally calculate: Accuracy, Precision

"""
# True Positives (tp):
tp = np.count_nonzero(np.logical_and(gtruth == 1, pred == 1))

# True Negatives (tn):
tn = np.count_nonzero(np.logical_and(gtruth == 0, pred == 0))

# False Positives (fp):
fp = np.count_nonzero(np.logical_and(gtruth == 0, pred == 1))

# False Negatives (fn):
fn = np.count_nonzero(np.logical_and(gtruth == 1, pred == 0))
"""

def compute_DICE_score(gtruth, pred):
    """Compute DICE score. Also known as F1 score.

    This method assumes gtruth and pred are binary. Label 1 is positive class and Label 0 is negative class.

    Parameters
    ----------
    gtruth : np.ndarray
        The binary ground truth segmentation for the considered image
    pred : np.ndarray
        The binary predicted segmentation for the considered image

    Returns
    ----------
    dice : float
        The dice score (== F1 score).
    """

    gtruth = np.asarray(gtruth)
    pred = np.asarray(pred)

    if np.max(gtruth) > 1:
        gtruth = gtruth / np.max(gtruth)

    if np.max(pred) > 1:
        pred = pred / np.max(pred)

    # Formula for Dice: dice = 2.0*(X⋂Y)/(|X|+|Y|)
    dice = 2.0 * np.count_nonzero(pred[gtruth == 1]) / (np.count_nonzero(gtruth) + np.count_nonzero(pred))

    return dice

def compute_IoU(gtruth, pred):
    """Compute IoU (Intersection-over-Union). IoU is also known as Jaccard Index.

    This method assumes gtruth and pred are binary. Label 1 is positive class and Label 0 is negative class.

    Parameters
    ----------
    gtruth : np.ndarray
        The binary ground truth segmentation for the considered image
    pred : np.ndarray
        The binary predicted segmentation for the considered image

    Returns
    ----------
    iou : float
        The IoU value (== Jaccard Index).
    """

    gtruth = np.asarray(gtruth)
    pred = np.asarray(pred)

    if np.max(gtruth) > 1:
        gtruth = gtruth / np.max(gtruth)
        gtruth.astype(np.uint)

    if np.max(pred) > 1:
        pred = pred / np.max(pred)

    # True Positives (tp):
    tp = np.count_nonzero(np.logical_and(gtruth == 1, pred == 1))

    # False Positives (fp):
    fp = np.count_nonzero(np.logical_and(gtruth == 0, pred == 1))

    # False Negatives (fn):
    fn = np.count_nonzero(np.logical_and(gtruth == 1, pred == 0))


    # Formula for IoU: iou = tp/(tp + fp + fn)
    iou = tp / (tp + fp + fn)

    return iou

def compute_specificity(gtruth, pred):
    """Compute specificity.

    This method assumes gtruth and pred are binary. Label 1 is positive class and Label 0 is negative class.

    Parameters
    ----------
    gtruth : np.ndarray
        The binary ground truth segmentation for the considered image
    pred : np.ndarray
        The binary predicted segmentation for the considered image

    Returns
    ----------
    sensitivity : float
        The specificity value.
    """

    gtruth = np.asarray(gtruth)
    pred = np.asarray(pred)

    if np.max(gtruth) > 1:
        gtruth = gtruth / np.max(gtruth)
        gtruth.astype(np.uint)

    if np.max(pred) > 1:
        pred = pred / np.max(pred)


    # True Negatives (tn):
    tn = np.count_nonzero(np.logical_and(gtruth == 0, pred == 0))

    # False Positives (fp):
    fp = np.count_nonzero(np.logical_and(gtruth == 0, pred == 1))

    if (tn + fp) == 0:
        print("skipped one specificity value as tn and fp are both 0")
        return -1

    # Formula for Specificity: specificity = tn/(tn + fp)
    specificity = tn / (tn + fp)

    return specificity

def compute_sensitivity(gtruth, pred):
    """Compute sensitivity. Also known as Recall.

    This method assumes gtruth and pred are binary. Label 1 is positive class and Label 0 is negative class.

    Parameters
    ----------
    gtruth : np.ndarray
        The binary ground truth segmentation for the considered image
    pred : np.ndarray
        The binary predicted segmentation for the considered image

    Returns
    ----------
    sensitivity : float
        The sensitivity value (== Recall).
    """

    gtruth = np.asarray(gtruth)
    pred = np.asarray(pred)

    if np.max(gtruth) > 1:
        gtruth = gtruth / np.max(gtruth)
        gtruth.astype(np.uint)

    if np.max(pred) > 1:
        pred = pred / np.max(pred)

    # True Positives (tp):
    tp = np.count_nonzero(np.logical_and(gtruth == 1, pred == 1))

    # False Negatives (fn):
    fn = np.count_nonzero(np.logical_and(gtruth == 1, pred == 0))

    if (tp + fn) == 0:
        print("skipped one sensitivity value as no tp and fn are both 0")
        return -1

    # Formula for Sensitivity: sensitivity = tp/(tp + fn)
    sensitivity = tp / (tp + fn)

    return sensitivity


def compute_precision(gtruth, pred):
    """Compute precision.

    This method assumes gtruth and pred are binary. Label 1 is positive class and Label 0 is negative class.

    Parameters
    ----------
    gtruth : np.ndarray
        The binary ground truth segmentation for the considered image
    pred : np.ndarray
        The binary predicted segmentation for the considered image

    Returns
    ----------
    precision : float
        The precision value.
    """

    gtruth = np.asarray(gtruth)
    pred = np.asarray(pred)

    if np.max(gtruth) > 1:
        gtruth = gtruth / np.max(gtruth)
        gtruth.astype(np.uint)

    if np.max(pred) > 1:
        pred = pred / np.max(pred)

    # True Positives (tp):
    tp = np.count_nonzero(np.logical_and(gtruth == 1, pred == 1))


    # False Positives (fp):
    fp = np.count_nonzero(np.logical_and(gtruth == 0, pred == 1))

    if (tp + fp) == 0:
        print("skipped one precision value as tp and fp are both 0")
        return -1

    # Formula for Precision: precision = tp/(tp + fp)
    precision = tp / (tp + fp)

    return precision


def compute_accuracy(gtruth, pred):
    """Compute accuracy.

    This method assumes gtruth and pred are binary. Label 1 is positive class and Label 0 is negative class.

    Parameters
    ----------
    gtruth : np.ndarray
        The binary ground truth segmentation for the considered image
    pred : np.ndarray
        The binary predicted segmentation for the considered image

    Returns
    ----------
    accuracy : float
        The accuracy value.
    """

    gtruth = np.asarray(gtruth)
    pred = np.asarray(pred)

    if np.max(gtruth) > 1:
        gtruth = gtruth / np.max(gtruth)
        gtruth.astype(np.uint)

    if np.max(pred) > 1:
        pred = pred / np.max(pred)

    # True Positives (tp):
    tp = np.count_nonzero(np.logical_and(gtruth == 1, pred == 1))

    # True Negatives (tn):
    tn = np.count_nonzero(np.logical_and(gtruth == 0, pred == 0))

    # False Positives (fp):
    fp = np.count_nonzero(np.logical_and(gtruth == 0, pred == 1))

    # False Negatives (fn):
    fn = np.count_nonzero(np.logical_and(gtruth == 1, pred == 0))


    # Formula for Accuracy: accuracy = (tp + tn)/(tp + tn + fp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy


def compute_metrics_on_folder(gtruth_folder, pred_folder):

    dsc_scores = []
    iou_scores = []
    accuracy_scores = []
    precision_scores = []
    specificity_scores = []
    sensitivity_scores = []

    gtruth_files = get_file_list_from_directory(gtruth_folder)
    pred_files = get_file_list_from_directory(pred_folder)

    num_gtruth_files = len(gtruth_files)
    num_pred_files = len(pred_files)

    assert num_pred_files == num_gtruth_files

    for i in range(num_gtruth_files):
        gtruth = read_image(gtruth_files[i])
        pred = read_image(pred_files[i])

        dsc = compute_DICE_score(gtruth, pred)
        iou = compute_IoU(gtruth, pred)
        accuracy = compute_accuracy(gtruth, pred)
        precision = compute_precision(gtruth, pred)
        specificity = compute_specificity(gtruth, pred)
        sensitivity = compute_sensitivity(gtruth, pred)

        if dsc != -1:
            dsc_scores.append(dsc)
        if iou != -1:
            iou_scores.append(iou)
        if accuracy != -1:
            accuracy_scores.append(accuracy)
        if precision != -1:
            precision_scores.append(precision)
        if specificity != -1:
            specificity_scores.append(specificity)
        if sensitivity != -1:
            sensitivity_scores.append(sensitivity)

    return np.asarray(dsc_scores), np.asarray(iou_scores), np.asarray(accuracy_scores), np.asarray(precision_scores), np.asarray(specificity_scores), np.asarray(sensitivity_scores)

def main():

    #gtruth_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\a1\gtruth"
    #pred_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\a1\pred"

    #gtruth_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\c2\gtruth"
    #pred_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\c2\pred"

    #gtruth_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\c4\gtruth"
    #pred_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\c4\pred"

    #gtruth_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\ml3b\gtruth"
    #pred_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\ml3b\pred"

    #gtruth_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\m5\gtruth"
    #pred_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\m5\pred"

    gtruth_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\ml3a\gtruth"
    pred_folder = r"E:\RESULTATE_UND_BILDER\FinalNN\the_evaluation\ml3a\pred"

    dice, iou, accuracy, precision, specificity, sensitivity = compute_metrics_on_folder(gtruth_folder, pred_folder)


    print("Mean Dice: ", np.mean(dice))
    print("StdDev Dice: ", np.std(dice))
    print("\n")
    print("Mean IoU: ", np.mean(iou))
    print("StdDev IoU: ", np.std(iou))
    print("\n")
    print("Mean Accuracy: ", np.mean(accuracy))
    print("StdDev Accuracy: ", np.std(accuracy))
    print("\n")
    print("Mean Precision: ", np.mean(precision))
    print("StdDev Precision: ", np.std(precision))
    print("\n")
    print("Mean Specificity: ", np.mean(specificity))
    print("StdDev Specificity: ", np.std(specificity))
    print("\n")
    print("Mean Sensitivity/Recall: ", np.mean(sensitivity))
    print("StdDev Sensitivity/Recall: ", np.std(sensitivity))


if __name__ == "__main__":
    main()
