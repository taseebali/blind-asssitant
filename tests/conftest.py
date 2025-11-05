"""Configuration for pytest"""

import pytest
import os
import sys

# Add the src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))