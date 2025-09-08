import os
import json
import re
from typing import Dict, List, Tuple


class FileService:
    """Service for managing files in the application"""

    def __init__(self):
        self.data_dir = "./app/data"
        self.config_file = "./app/config/settings.conf"
        self.special_datasets = {
            "mnist": {
                "description": "MNIST handwritten digits dataset (28x28 grayscale images)",
                "type": "image",
                "dir": "./app/data/MNIST",
            }
        }

    def get_available_files(self) -> Dict[str, List]:
        """
        Get all available data files categorized by type (csv, image, special)

        Returns:
            Dict with keys: 'csv', 'image', and 'special', each containing a list of file paths or dataset info
        """
        files = {"csv": [], "image": [], "special": []}

        # Walk through the data directory
        for root, dirs, filenames in os.walk(self.data_dir):
            # Skip special dataset dirs that we handle separately
            if any(
                special_dir in root
                for special_dir in [os.path.join(self.data_dir, "MNIST")]
            ):
                continue

            for filename in filenames:
                # Skip __init__.py and other non-data files
                if filename.startswith("__") or filename.startswith("."):
                    continue

                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(
                    full_path, start=os.path.dirname(self.data_dir)
                )

                # Categorize by extension
                if filename.lower().endswith(".csv"):
                    files["csv"].append(rel_path)

        # Add special datasets
        for name, info in self.special_datasets.items():
            if os.path.exists(info["dir"]):
                files["special"].append(
                    {
                        "name": name,
                        "description": info["description"],
                        "type": info["type"],
                    }
                )

        return files

    def set_selected_dataset(self, file_path: str, has_header: bool) -> Dict[str, str]:
        """
        Update the settings.conf file with the selected dataset

        Args:
            file_path: Path to the selected data file or special dataset name
            has_header: Whether the dataset has a header row

        Returns:
            Dict with the update status
        """
        # Check if it's a special dataset
        if file_path in self.special_datasets:
            dataset_path = file_path
        else:
            # Regular file validation
            # Always use forward slashes for cross-platform compatibility in JSON
            if not file_path.startswith("./app"):
                dataset_path = ("./app/" + file_path.lstrip("/")).replace("\\", "/")
            else:
                dataset_path = file_path.replace("\\", "/")

            # Convert to OS-specific path for file check only
            os_specific_path = os.path.normpath(dataset_path)
            if not os.path.isfile(os_specific_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        # Update settings in proper JSON format
        with open(self.config_file, "w") as f:
            f.write("{\n")
            f.write(f'    "data_file": "{dataset_path}",\n')
            # For the last property, no comma at the end
            f.write(f'    "has_header": {str(has_header).lower()}\n')
            f.write("}\n")

        return {
            "message": f"Dataset {file_path} selected successfully",
            "file": file_path,
        }

    def get_current_dataset(self) -> Dict[str, str]:
        """
        Get the currently selected dataset from settings.conf

        Returns:
            Dict with the current dataset file path or name
        """
        try:
            with open(self.config_file, "r") as f:
                content = f.read().strip()
                if content:
                    # Parse the setting manually since it's not standard JSON
                    match = re.search(r'"data_file":\s*"([^"]+)"', content)
                    if match:
                        data_file = match.group(1)
                        # Check if it's a special dataset
                        if data_file in self.special_datasets:
                            return {
                                "data_file": data_file,
                                "type": "special",
                                "description": self.special_datasets[data_file][
                                    "description"
                                ],
                            }
                        return {"data_file": data_file, "type": "file"}
                return {"data_file": "", "type": ""}
        except FileNotFoundError:
            return {"data_file": "", "type": ""}
