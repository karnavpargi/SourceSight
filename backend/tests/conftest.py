import os
import sys
from pathlib import Path

# Unit tests mock the LLM; satisfy chat provider config validation at import time.
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key-for-unit-tests")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
