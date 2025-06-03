import os
import shutil
from datetime import datetime
from typing import Optional

class ModelSaver:
    """
    Handles saving and packaging trained models with their logs and artifacts.
    
    Args:
        model_name: Name of the model (used for naming files/folders)
        root_dir: Root directory where files will be saved (defaults to current directory)
    """
    
    def __init__(self, model_name: str, root_dir: str = "."):
        self.model_name = model_name
        self.root_dir = os.path.abspath(root_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    def save_model(
        self,
        model_path: str,
        logs_path: Optional[str] = None,
        additional_files: Optional[list] = None,
        create_zip: bool = True
    ) -> str:
        """
        Saves model with all related files.
        
        Args:
            model_path: Path to the model file (.pth or similar)
            logs_path: Path to training logs directory (optional)
            additional_files: List of additional files to include (optional)
            create_zip: Whether to create a zip archive of all files
            
        Returns:
            Path to the saved model or zip file
        """
        # Create output directory structure
        output_folder = self._create_output_folder()
        
        # Copy model file
        self._copy_model_file(model_path, output_folder)
        
        # Copy logs if provided
        if logs_path:
            self._copy_logs(logs_path, output_folder)
            
        # Copy additional files if provided
        if additional_files:
            self._copy_additional_files(additional_files, output_folder)
            
        # Create zip archive if requested
        if create_zip:
            return self._create_zip_archive(output_folder)
            
        return output_folder
    
    def _create_output_folder(self) -> str:
        """Creates the output folder with timestamp"""
        folder_name = f"{self.model_name}_{self.timestamp}"
        output_folder = os.path.join(self.root_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        return output_folder
    
    def _copy_model_file(self, src_path: str, dest_folder: str):
        """Copies the model file to destination folder"""
        dest_path = os.path.join(dest_folder, f"{self.model_name}_model.pth")
        shutil.copy(src_path, dest_path)
    
    def _copy_logs(self, src_path: str, dest_folder: str):
        """Copies logs directory to destination folder"""
        dest_path = os.path.join(dest_folder, "logs")
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)
    
    def _copy_additional_files(self, files: list, dest_folder: str):
        """Copies additional files to destination folder"""
        for file_path in files:
            if os.path.exists(file_path):
                shutil.copy(file_path, dest_folder)
    
    def _create_zip_archive(self, folder_path: str) -> str:
        """Creates a zip archive of the folder"""
        zip_filename = f"{os.path.basename(folder_path)}.zip"
        zip_path = os.path.join(self.root_dir, zip_filename)
        
        # Remove existing zip if present
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        # Create new zip archive
        shutil.make_archive(
            base_name=zip_path.replace(".zip", ""),
            format="zip",
            root_dir=folder_path,
            base_dir="."
        )
        
        return zip_path


# Example usage
if __name__ == "__main__":
    # Initialize saver
    saver = ModelSaver(
        model_name="lstm_heavy",
        root_dir="./saved_models"  # Save to local 'saved_models' directory
    )
    
    # Save the model with all artifacts
    result_path = saver.save_model(
        model_path="./best_model.pth",
        logs_path="./training_logs",
        additional_files=["./config.yaml", "./results.csv"]
    )
    
    print(f"Model saved to: {result_path}")