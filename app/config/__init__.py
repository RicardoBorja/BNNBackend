import os
import json
import re
from typing import Dict, Any, Optional


class Configuration:
    """Configuration class to handle reading settings from settings.conf file"""

    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(__file__), "settings.conf")
        self.settings = {}
        self.load_config()

    def load_config(self):
        """Load configuration from settings.conf file"""
        try:
            with open(self.config_file, "r") as f:
                content = f.read().strip()
                if content:
                    # Try to parse as JSON first with some preprocessing
                    try:
                        # Replace Python True/False with JSON true/false
                        content = content.replace("True", "true").replace(
                            "False", "false"
                        )
                        # Remove trailing commas which are invalid in JSON
                        content = re.sub(r",\s*}", "}", content)
                        content = re.sub(r",\s*]", "]", content)

                        # Parse as JSON
                        self.settings = json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback to manual parsing if JSON parsing fails
                        self._parse_settings(content)
        except FileNotFoundError:
            # Use default values if file doesn't exist
            self.settings = {
                "data_file": "./app/data/pima-indians-diabetes.data.csv",
                "has_header": False,
            }

    def _parse_settings(self, content: str):
        """Parse the config file content into a dictionary

        Args:
            content (str): Content of the settings.conf file
        """
        # Reset settings
        self.settings = {}

        # Remove curly braces if present
        content = content.strip()
        if content.startswith("{"):
            content = content[1:]
        if content.endswith("}"):
            content = content[:-1]

        # Process each line
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Remove trailing comma if present
            if line.endswith(","):
                line = line[:-1]

            # Extract key and value
            match = re.match(r'"([^"]+)":\s*(.*)', line)
            if match:
                key = match.group(1)
                value_str = match.group(2).strip()

                # Parse the value based on its format
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                elif value_str.startswith('"') and value_str.endswith('"'):
                    value = value_str[1:-1]  # Remove quotes
                elif value_str.isdigit():
                    value = int(value_str)
                elif re.match(r"^-?\d+(\.\d+)?$", value_str):
                    value = float(value_str)
                else:
                    value = value_str

                self.settings[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key

        Args:
            key (str): The configuration key to retrieve
            default (Any, optional): Default value if key is not found

        Returns:
            Any: The configuration value
        """
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value

        Args:
            key (str): The configuration key to set
            value (Any): The value to set
        """
        self.settings[key] = value
        self._save_config()

    def _save_config(self) -> None:
        """Save the current configuration to the settings.conf file"""
        try:
            # Format settings in the exact required format
            with open(self.config_file, "w") as f:
                f.write("{\n")

                # Get all keys and determine the last one for comma handling
                keys = list(self.settings.keys())

                for i, key in enumerate(keys):
                    value = self.settings[key]
                    is_last_item = i == len(keys) - 1

                    # Format the value based on its type
                    if isinstance(value, bool):
                        formatted_value = str(value).lower()
                    elif isinstance(value, str):
                        formatted_value = f'"{value}"'
                    else:
                        formatted_value = str(value)

                    # Add comma for all items except the last
                    comma = ",\n" if not is_last_item else "\n"
                    f.write(f'    "{key}": {formatted_value}{comma}')

                f.write("}\n")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values

        Returns:
            Dict[str, Any]: Dictionary of all configuration values
        """
        return self.settings.copy()


# Create a singleton instance named 'conf' for easy access
conf = Configuration()
