import numpy as np
from numpy.typing import NDArray
from tensorflow.keras import models


def dwis_classifier(dwis: NDArray, threshold: float) -> NDArray:
    """Classify DWIs as good or bad using a pre-trained CNN.

    Args:
        dwis (NDArray): Image stack to classify with shape
            ``(n_images, rows, cols)``.
        threshold (float): Probability threshold. Predictions below this value
            are labelled bad (``0``); predictions at or above are labelled
            good (``1``).

    Returns:
        NDArray: Label vector of shape ``(n_images, 1)`` where ``0`` means bad
        and ``1`` means good.
    """

    cnn_name = "/usr/local/dtcmr/dwi_classifier/classifier_dwis.hdf5"
    # load the classifier model
    model = models.load_model(cnn_name)

    # normalise images
    dwis = np.float32(dwis)
    dwis *= 1 / dwis.max()

    # run the model predictions for all images
    dwis = np.expand_dims(dwis, -1)
    prob_label = model.predict(dwis)

    # round to 0 or 1 according to threshold value
    predicted_label = prob_label.copy()
    predicted_label[predicted_label < threshold] = 0
    predicted_label[predicted_label >= threshold] = 1

    return predicted_label


if __name__ == "__main__":
    dwis = np.random.rand(10, 256, 96)
    labels = dwis_classifier(dwis)
    print(labels.shape)
