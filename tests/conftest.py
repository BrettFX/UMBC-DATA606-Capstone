"""Test-wide setup: hide the GPU so tests never touch it (a long annotation or training job may own it)."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
