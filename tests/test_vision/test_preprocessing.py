import numpy as np

def test_load_and_resize_image(preprocessor, sample_synthetic_image):
    # Test with numpy array
    loaded = preprocessor.load_image(sample_synthetic_image)
    resized = preprocessor.resize(loaded)
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8

def test_grayscale_and_threshold(preprocessor, sample_synthetic_image):
    thresh = preprocessor.threshold_otsu(sample_synthetic_image)
    assert len(thresh.shape) == 2
    assert thresh.shape == (224, 224)
    # Check that values are either 0 or 255
    unique_vals = np.unique(thresh)
    for val in unique_vals:
        assert val in [0, 255]
