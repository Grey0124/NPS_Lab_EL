#!/usr/bin/env python3
"""
Standalone Registry Service for ARP Detection Web Application
"""

import logging
from typing import Dict, Any, Optional
from .arp_registry import ARPRegistry

logger = logging.getLogger(__name__)

class RegistryService:
    """Service for managing ARP registry independently of the detector."""
    
    def __init__(self, registry_path: str = "data/registry.yml"):
        self.registry_manager = ARPRegistry(registry_path)
    
    async def get_entries(self) -> Dict[str, str]:
        """Get all registry entries."""
        try:
            return self.registry_manager.list_entries()
        except Exception as e:
            logger.error(f"Error getting registry entries: {e}")
            return {}
    
    async def add_entry(self, ip: str, mac: str) -> bool:
        """Add a registry entry."""
        try:
            return self.registry_manager.add_entry(ip, mac)
        except Exception as e:
            logger.error(f"Error adding registry entry: {e}")
            return False
    
    async def remove_entry(self, ip: str) -> bool:
        """Remove a registry entry."""
        try:
            return self.registry_manager.remove_entry(ip)
        except Exception as e:
            logger.error(f"Error removing registry entry: {e}")
            return False
    
    async def reset_registry(self) -> bool:
        """Reset the registry."""
        try:
            self.registry_manager.reset()
            return True
        except Exception as e:
            logger.error(f"Error resetting registry: {e}")
            return False
    
    async def import_entries(self, entries: Dict[str, str]) -> int:
        """Import multiple entries."""
        try:
            # Clear existing entries
            self.registry_manager.reset()
            
            # Add new entries
            added_count = 0
            for ip, mac in entries.items():
                if self.registry_manager.add_entry(ip, mac):
                    added_count += 1
            
            return added_count
        except Exception as e:
            logger.error(f"Error importing registry entries: {e}")
            return 0
    
    def get_registry_path(self) -> str:
        """Get the registry file path."""
        return str(self.registry_manager.registry_path) 