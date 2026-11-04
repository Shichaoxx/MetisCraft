import os

# Tests run on CPU with tiny random models; never touch the shared GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
