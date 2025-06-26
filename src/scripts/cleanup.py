import os
import shutil
from src.config import SKELETONS_DIR, PROTOTYPES_DIR, VISUALIZATIONS_DIR

# Clears generated files
def clean():
    # List of directories whose contents will be deleted
    dirs_to_clean = [
        SKELETONS_DIR,
        PROTOTYPES_DIR,
        VISUALIZATIONS_DIR
    ]
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            for item_name in os.listdir(directory):
                item_path = os.path.join(directory, item_name)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                except Exception as e:
                    print(f"Error deleting {item_path}: {e}")

    print("Cleanup complete!")

if __name__ == "__main__":
    clean()