#!/usr/bin/env python3
"""
Configuration Service for ARP Detection Web Application
"""

import os
import yaml
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class ConfigService:
    """Service for managing application configuration."""
    
    def __init__(self, config_path: str = "data/config.yml"):
        self.config_path = Path(config_path)
        self.backup_dir = Path("data/backups")
        self.config = self._load_default_config()
        self._ensure_directories()
        self._load_config()
    
    def _ensure_directories(self):
        """Ensure config and backup directories exist."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
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
            },
            'system': {
                'auto_start': False,
                'log_level': 'INFO',
                'max_log_size': 100,
                'enable_backup': True,
                'backup_interval': 24
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
    
    async def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self.config.get('system', {})
    
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
    
    async def update_system_config(self, config: Dict[str, Any]) -> bool:
        """Update system configuration."""
        try:
            self.config['system'].update(config)
            self._save_config()
            logger.info("System configuration updated")
            return True
        except Exception as e:
            logger.error(f"Error updating system config: {e}")
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
    
    async def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        errors = []
        warnings = []
        
        try:
            # Validate detection config
            detection = self.config.get('detection', {})
            if not detection.get('registry_path'):
                errors.append("Registry path is required")
            elif not Path(detection['registry_path']).exists():
                warnings.append(f"Registry file not found: {detection['registry_path']}")
            
            if not detection.get('model_path'):
                errors.append("Model path is required")
            elif not Path(detection['model_path']).exists():
                warnings.append(f"Model file not found: {detection['model_path']}")
            
            threshold = detection.get('detection_threshold')
            if threshold is not None and (threshold < 0 or threshold > 1):
                errors.append("Detection threshold must be between 0 and 1")
            
            # Validate alerts config
            alerts = self.config.get('alerts', {})
            if alerts.get('email_enabled') and not alerts.get('email_recipients'):
                warnings.append("Email alerts enabled but no recipients specified")
            
            if alerts.get('webhook_enabled') and not alerts.get('webhook_url'):
                errors.append("Webhook alerts enabled but no URL specified")
            
            # Validate web config
            web = self.config.get('web', {})
            port = web.get('port')
            if port and (port < 1 or port > 65535):
                errors.append("Web port must be between 1 and 65535")
            
            # Validate logging config
            logging_config = self.config.get('logging', {})
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if logging_config.get('level') not in valid_levels:
                errors.append(f"Log level must be one of: {', '.join(valid_levels)}")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    async def import_config(self, config_data: Dict[str, Any]) -> bool:
        """Import configuration from dictionary."""
        try:
            # Validate the imported config
            validation_result = await self._validate_import_config(config_data)
            if not validation_result["valid"]:
                logger.error(f"Invalid configuration import: {validation_result['errors']}")
                return False
            
            # Merge with current config
            self._merge_config(self.config, config_data)
            self._save_config()
            logger.info("Configuration imported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error importing config: {e}")
            return False
    
    async def _validate_import_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate imported configuration."""
        errors = []
        
        # Check for required sections
        required_sections = ['detection', 'alerts', 'web', 'database', 'logging']
        for section in required_sections:
            if section not in config_data:
                errors.append(f"Missing required section: {section}")
        
        # Validate detection section
        if 'detection' in config_data:
            detection = config_data['detection']
            if not isinstance(detection.get('detection_threshold'), (int, float)):
                errors.append("Detection threshold must be a number")
            elif detection.get('detection_threshold', 0) < 0 or detection.get('detection_threshold', 0) > 1:
                errors.append("Detection threshold must be between 0 and 1")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def create_backup(self) -> str:
        """Create a backup of current configuration."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}.yml"
            backup_path = self.backup_dir / backup_name
            
            # Copy current config
            shutil.copy2(self.config_path, backup_path)
            
            # Create metadata
            metadata = {
                "backup_created": datetime.now().isoformat(),
                "original_path": str(self.config_path),
                "version": "1.0.0"
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Configuration backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List available configuration backups."""
        try:
            backups = []
            for backup_file in self.backup_dir.glob("config_backup_*.yml"):
                metadata_file = backup_file.with_suffix('.json')
                metadata = {}
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error reading backup metadata: {e}")
                
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "created": metadata.get("backup_created", "Unknown"),
                    "size": backup_file.stat().st_size
                })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    async def restore_backup(self, backup_name: str) -> bool:
        """Restore configuration from backup."""
        try:
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_path}")
                return False
            
            # Create backup of current config before restoring
            await self.create_backup()
            
            # Restore the backup
            shutil.copy2(backup_path, self.config_path)
            
            # Reload the configuration
            self._load_config()
            
            logger.info(f"Configuration restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def get_config_path(self) -> str:
        """Get configuration file path."""
        return str(self.config_path) 