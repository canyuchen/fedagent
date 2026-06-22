"""Session state I/O.

Extracted from FederatedServer. Pure read/write of session_state.json and
enumeration of available sessions. State binding (which fields to
persist/restore) stays with FederatedServer — this module only moves bytes.
"""

import json
from pathlib import Path
from typing import List, Optional


class SessionManager:
    STATE_FILENAME = "session_state.json"
    SESSION_PREFIX = "federated_session_"

    def __init__(self, output_dir, logger):
        self.output_dir = Path(output_dir)
        self.logger = logger

    def save(self, state: dict):
        state_file = self.output_dir / self.STATE_FILENAME
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        self.logger.info(f"Session state saved to {state_file}")

    def load(self) -> Optional[dict]:
        state_file = self.output_dir / self.STATE_FILENAME
        if not state_file.exists():
            self.logger.warning(f"Session state file not found: {state_file}")
            return None
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load session state: {str(e)}")
            return None

    def list_sessions(self, base_output_dir: Path) -> List[str]:
        base = Path(base_output_dir)
        if not base.exists():
            return []
        return sorted(
            item.name for item in base.iterdir()
            if item.is_dir() and item.name.startswith(self.SESSION_PREFIX)
        )
