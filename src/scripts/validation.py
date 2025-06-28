# src/scripts/validation.py
import os
import time
import matplotlib.pyplot as plt
from src.config import VALIDATION_DATA_DIR, SIMILARITY_THRESHOLD, REPORTS_DIR
from .prediction import get_prediction 

def generate_accuracy_plot(results):
    """Generates and saves a bar chart of the accuracy results."""
    pose_names = list(results.keys())
    accuracies = [res['accuracy'] for res in results.values()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(pose_names, accuracies, color='skyblue')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy per Pose')
    ax.set_ylim(0, 100)
    
    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_path = os.path.join(REPORTS_DIR, "accuracy_report.png")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nAccuracy report graph saved to '{output_path}'")


def validate_all():
    """
    Runs prediction on the validation set, collects stats,
    prints a report, and generates an accuracy graph.
    """
    if not os.path.exists(VALIDATION_DATA_DIR):
        print(f"Validation directory not found at: '{VALIDATION_DATA_DIR}'")
        return

    print("--- Starting Full Validation Process ---")
    
    stats = {}

    for true_pose_class in sorted(os.listdir(VALIDATION_DATA_DIR)):
        class_dir = os.path.join(VALIDATION_DATA_DIR, true_pose_class)
        if not os.path.isdir(class_dir):
            continue
        
        stats[true_pose_class] = {'correct': 0, 'total': 0, 'accuracy': 0.0}
        image_files = [f for f in sorted(os.listdir(class_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for filename in image_files:
            image_path = os.path.join(class_dir, filename)
            print(f"\n--- Validating: {true_pose_class}/{filename} ---")
            
            predicted_pose, similarity, error = get_prediction(image_path)
            stats[true_pose_class]['total'] += 1

            if error:
                print(f"Prediction failed: {error}")
                continue
            
            print(f"Prediction: '{predicted_pose}' (Similarity: {similarity:.2%})")

            # Check if the prediction is correct
            if predicted_pose == true_pose_class and similarity >= SIMILARITY_THRESHOLD:
                print("Result: CORRECT")
                stats[true_pose_class]['correct'] += 1
            else:
                print("Result: INCORRECT")
            
            time.sleep(0.5)

    # --- Generate Final Report ---
    print("\n\n--- VALIDATION SUMMARY ---")
    total_correct = 0
    total_images = 0
    
    for pose, result in stats.items():
        if result['total'] > 0:
            accuracy = (result['correct'] / result['total']) * 100
            result['accuracy'] = accuracy
            total_correct += result['correct']
            total_images += result['total']
            print(f"- {pose}: {result['correct']}/{result['total']} correct ({accuracy:.1f}%)")
    
    if total_images > 0:
        overall_accuracy = (total_correct / total_images) * 100
        print(f"\nOverall Accuracy: {total_correct}/{total_images} correct ({overall_accuracy:.1f}%)")
        
        # Generate and save the plot
        generate_accuracy_plot(stats)
    else:
        print("No images found to validate.")
            
    print("\n--- Validation Process Finished ---")