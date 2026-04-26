import sys
import os

_generated_dir = os.path.dirname(__file__)
if _generated_dir not in sys.path:
    sys.path.insert(0, _generated_dir)

from .secure_nlp_pb2 import *
from .secure_nlp_pb2_grpc import *
from .local_client import TEEClient