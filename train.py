# train.py

import os
import pickle
import numpy as np
from config import TRAINING_DATA_DIR, PROTOTYPE_FILE
from pose_extractor import extract_landmarks

def train_model():
    """
    Processa as imagens de treinamento e cria um vetor protótipo da pose.
    """
    landmark_vectors = []
    
    print(f"Iniciando treinamento com as imagens da pasta: '{TRAINING_DATA_DIR}'...")

    image_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Erro: Nenhuma imagem encontrada em '{TRAINING_DATA_DIR}'.")
        return

    for filename in image_files:
        image_path = os.path.join(TRAINING_DATA_DIR, filename)
        print(f"Processando imagem: {filename}")
        
        landmarks = extract_landmarks(image_path)
        if landmarks is not None:
            landmark_vectors.append(landmarks)
        else:
            print(f"  -> Aviso: Nenhuma pose detectada em {filename}.")
    
    if not landmark_vectors:
        print("Treinamento falhou. Nenhuma pose foi detectada em nenhuma das imagens de treino.")
        return

    # Calcula o vetor "médio" de todos os landmarks de treino
    prototype_vector = np.mean(landmark_vectors, axis=0)
    
    # Normaliza o protótipo final
    prototype_vector /= np.linalg.norm(prototype_vector)

    # Salva o protótipo em um arquivo
    with open(PROTOTYPE_FILE, 'wb') as f:
        pickle.dump(prototype_vector, f)

    print("\nTreinamento concluído com sucesso!")
    print(f"Vetor protótipo da pose 'Double Biceps' foi salvo em '{PROTOTYPE_FILE}'.")


if __name__ == "__main__":
    train_model()