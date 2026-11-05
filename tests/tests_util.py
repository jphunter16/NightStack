import os
import tempfile

from make_synthetic import make


def ensure_data():
    d = os.environ.get("NIGHTSTACK_TEST_DATA") or os.path.join(tempfile.gettempdir(), "nightstack-synthetic", "frames")
    if not os.path.exists(os.path.join(d, "truth_H.npy")):
        make(d)
    return d
