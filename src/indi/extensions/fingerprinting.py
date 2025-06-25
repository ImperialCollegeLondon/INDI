import os
from collections import namedtuple

import numpy as np
from tqdm import tqdm


def model(S0, g, b, D):
    attenuation = -b * np.einsum("bi,ij,bj->b", g, D, g)
    return S0 * np.exp(attenuation)


def generate_dictionary(b_values, diffusion_directions):

    n_tensors = 50000
    Ds = np.zeros((n_tensors, 3, 3))

    signals = np.zeros((n_tensors, len(b_values)))
    Ds = np.zeros((n_tensors, 3, 3))
    bar = tqdm(total=n_tensors, desc="Generating dictionary", unit="tensor")
    for i in range(n_tensors):
        eigenvalues = np.random.uniform(0.0001, 0.01, 3)
        eigenvectors = np.random.randn(3, 3)
        eigenvectors, _ = np.linalg.qr(eigenvectors)  # Ensure orthogonality
        D = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        Ds[i] = D
        signals[i] = model(1.0, diffusion_directions, b_values, D)
        bar.update(1)
    bar.close()

    return signals, Ds


def compress_dictionary(dictionary, rank=10):
    _, _, Vt = np.linalg.svd(dictionary, full_matrices=False)
    dictionary_compressed = dictionary @ Vt[:rank, :].T
    return dictionary_compressed, Vt[:rank, :]


def dictionary_matching_bf(image_data, dictionary, logger):
    dictionary, Vt = compress_dictionary(dictionary, rank=10)
    logger.info(f"Compressed dictionary to shape {dictionary.shape}")
    indices = np.argmax(np.squeeze(dictionary[np.newaxis, :, :] @ (Vt[:10, :] @ image_data.T)), axis=0)
    logger.info(f"Found indices of best matches in dictionary: {indices}")
    return indices


def dictionary_matching_hnsw(image_data, dictionary, settings, logger):
    import hnswlib

    if not os.path.exists(os.path.join(settings["session"], "index.bin")):
        logger.info("Creating new HNSW index")
        p = hnswlib.Index(space="cosine", dim=image_data.shape[-1])
        p.init_index(max_elements=dictionary.shape[0], ef_construction=20, M=1)
        logger.info(f"Initialized HNSW index with {dictionary.shape[0]} elements")
        p.add_items(dictionary, list(range(dictionary.shape[0])))
        p.set_ef(10)
        p.save_index(os.path.join(settings["session"], "index.bin"))
    else:
        logger.info("Loading existing HNSW index")
        p = hnswlib.Index(space="cosine", dim=image_data.shape[-1])
        p.load_index(os.path.join(settings["session"], "index.bin"))
    logger.info("Added items to HNSW index")
    indices, _ = p.knn_query(image_data, k=1)
    logger.info("Performed k-NN query on HNSW index")
    return indices.flatten()


def fingerprinting_fit(image_data, bvals, bvecs, settings, logger):

    vmax, vmin = image_data.max(), image_data.min()
    image_data = (image_data - vmin) / (vmax - vmin)

    if os.path.exists(os.path.join(settings["session"], "fingerprinting_dictionary.npz")):
        logger.info("Loading precomputed dictionary")
        data = np.load(os.path.join(settings["session"], "fingerprinting_dictionary.npz"))
        dictionary = data["dictionary"]
        Ds = data["Ds"]
    else:
        logger.info("Generating dictionary for fingerprinting")
        dictionary, Ds = generate_dictionary(bvals, bvecs)
        np.savez(os.path.join(settings["session"], "fingerprinting_dictionary.npz"), dictionary=dictionary, Ds=Ds)

    logger.info(f"Generated dictionary with shape {dictionary.shape}")
    image_shape = image_data.shape[:2]
    image_data = image_data.reshape(-1, image_data.shape[-1])
    image_data_mag = np.linalg.norm(image_data, axis=1, keepdims=True)
    dictionary_mag = np.linalg.norm(dictionary, axis=1, keepdims=True)
    indices = dictionary_matching_bf(image_data / image_data_mag, dictionary / dictionary_mag, logger)
    # indices = dictionary_matching_hnsw(image_data, dictionary, settings, logger)

    D_est = Ds[indices]
    S0_est = np.squeeze(image_data_mag) / np.squeeze(dictionary_mag[indices])
    logger.info(f"Estimated D shape: {D_est.shape}, Estimated S0 shape: {S0_est.shape}")
    tenfit = namedtuple("TensorFit", ["quadratic_form", "S0_hat"])
    tenfit.quadratic_form = D_est.reshape(*image_shape, 3, 3)
    tenfit.S0_hat = S0_est.reshape(*image_shape)
    return tenfit
