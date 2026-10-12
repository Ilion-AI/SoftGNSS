"""Allow running the GNSS receiver as a module: python -m gnss_receiver"""

from .main import main
import sys

sys.exit(main())
