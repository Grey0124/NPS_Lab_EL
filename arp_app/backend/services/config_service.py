#!/usr/bin/env python3
"""
Configuration Service for ARP Detection Web Application
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigService:
    """Service for managing application configuration."""
    
    def __init__(self, config_path: str = "data/config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        self._ensure_config_directory()
        self._load_config()
    
    def _ensure_config_directory(self):
        """Ensure config directory exists."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'detection': {
                'registry_path': 'data/registry.yml',
                'model_path': 'models/realistic_rf_model.joblib',
                'log_file': 'logs/arp_detector.log',
                'detection_threshold': 0.7,
                'batch_size': 100,
                'max_queue_size': 1000,
                'alert_cooldown': 300
            },
            'alerts': {
                'email_enabled': False,
                'email_recipients': [],
                'webhook_enabled': False,
                'webhook_url': None,
                'notification_cooldown': 300
            },
            'web': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': True,
                'cors_origins': ['http://localhost:5173', 'http://localhost:3000']
            },
            'database': {
                'url': 'sqlite:///data/arp_detector.db',
                'echo': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/app.log'
            }
        }
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                
                # Merge with defaults
                self._merge_config(self.config, file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self._save_config()
                logger.info(f"Created default configuration at {self.config_path}")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    async def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.config.get('detection', {})
    
    async def get_alert_config(self) -> Dict[str, Any]:
        """Get alert configuration."""
        return self.config.get('alerts', {})
    
    async def get_web_config(self) -> Dict[str, Any]:
        """Get web server configuration."""
        return self.config.get('web', {})
    
    async def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})
    
    async def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    async def update_detection_config(self, config: Dict[str, Any]) -> bool:
        """Update detection configuration."""
        try:
            self.config['detection'].update(config)
            self._save_config()
            logger.info("Detection configuration updated")
            return True
        except Exception as e:
            logger.error(f"Error updating detection config: {e}")
            return False
    
    async def update_alert_config(self, config: Dict[str, Any]) -> bool:
        """Update alert configuration."""
        try:
            self.config['alerts'].update(config)
            self._save_config()
            logger.info("Alert configuration updated")
            return True
        except Exception as e:
            logger.error(f"Error updating alert config: {e}")
            return False
    
    async def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self.config.copy()
    
    async def reset_config(self) -> bool:
        """Reset configuration to defaults."""
        try:
            self.config = self._load_default_config()
            self._save_config()
            logger.info("Configuration reset to defaults")
            return True
        except Exception as e:
            logger.error(f"Error resetting config: {e}")
            return False
    
    def get_config_path(self) -> str:
        """Get configuration file path."""
        return str(self.config_path) 