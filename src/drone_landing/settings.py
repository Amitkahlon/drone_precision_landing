"""Paths and simulation-wide defaults.

Control gains live next to the controller in :mod:`drone_landing.control.gains`;
everything here is about where files are and how long a run is allowed to take.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

MODEL_PATH = PACKAGE_DIR / "assets" / "drone.xml"

# Results are written relative to the working directory so a batch lands next to
# wherever it was launched from, rather than inside the installed package.
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_RUNS_FILE = DEFAULT_RESULTS_DIR / "runs.jsonl"

DEFAULT_HOVER_ALTITUDE_M = 3.0
DEFAULT_TIMEOUT_S = 120.0

# The floor plane is 20x20 m, so past this the drone can no longer recover.
WORLD_LIMIT_M = 25.0

# Attempts to place a waypoint at the requested spacing before taking the best
# candidate found so far.
WAYPOINT_SPACING_ATTEMPTS = 200

# Buckets used when grouping batch results by mean leg speed.
SPEED_BUCKET_COUNT = 4
