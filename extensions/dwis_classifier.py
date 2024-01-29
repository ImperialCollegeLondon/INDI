import numpy as np
from numpy.typing import NDArray
from tensorflow.keras import models


def dwis_classifier(dwis: NDArray, threshold: float) -> NDArray:
    """
    Classify DWIs as either good (0) or bad (1)

    Parameters
    ----------
    dwis : numpy array
        array with dwis to be classified
    threshold : float, optional
        threshold value. If lower than threshold, the image is labeled as bad.

    Returns
    -------
    numpy array
        vector with labels for each dwi
        (o = bad, 1 = good)
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
