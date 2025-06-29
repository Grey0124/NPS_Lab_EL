import os
import yaml
from pathlib import Path
from typing import Dict, Optional

REGISTRY_PATH = Path('data/registry.yml')

class ARPRegistry:
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = Path(registry_path) if registry_path else REGISTRY_PATH
        self._ensure_registry_exists()
        self.entries = self._load_registry()

    def _ensure_registry_exists(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            with open(self.registry_path, 'w') as f:
                yaml.dump({}, f)

    def _load_registry(self) -> Dict[str, str]:
        if self.registry_path.exists():
            entries = {}
            try:
                with open(self.registry_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and ':' in line and not line.startswith('#'):
                            # Parse simple "key: value" format
                            parts = line.split(':', 1)  # Split on first colon only
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                if key and value:
                                    entries[key] = value
                return entries
            except Exception as e:
                print(f"Warning: Error loading registry file: {e}")
                return {}
        return {}

    def reload(self):
        """Reload entries from the registry file."""
        self.entries = self._load_registry()

    def save(self):
        with open(self.registry_path, 'w') as f:
            for ip, mac in self.entries.items():
                f.write(f"{ip}: {mac}\n")

    def add_entry(self, ip: str, mac: str) -> bool:
        # Reload from file to ensure we have the latest data
        self.reload()
        
        if ip in self.entries and self.entries[ip] == mac:
            return False  # Already present
        self.entries[ip] = mac
        self.save()
        return True

    def remove_entry(self, ip: str) -> bool:
        if ip in self.entries:
            del self.entries[ip]
            self.save()
            return True
        return False

    def get_mac(self, ip: str) -> Optional[str]:
        return self.entries.get(ip)

    def list_entries(self) -> Dict[str, str]:
        return dict(self.entries)

    def reset(self):
        self.entries = {}
        self.save() 