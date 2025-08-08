# Ensure tests import modules from this service directory first.
# This avoids import collisions with other `app/*` packages in the monorepo
# and makes `import app.server` and `from app import ...` behave consistently.
import os
import sys

SERVICE_ROOT = os.path.dirname(__file__)
APP_DIR = os.path.join(SERVICE_ROOT, "app")

# Prepend both the service root (so `import app.*` works) and the app dir itself
# (so local-style imports like `from oauth.decorator import ...` still work).
for p in (SERVICE_ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
