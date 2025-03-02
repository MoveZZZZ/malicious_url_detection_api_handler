from CustomModels import DeepMLP_3, DeepMLP_5, DeepMLP_7, AutoencoderClassifier
from tensorflow.keras.models import load_model
import os
import joblib

class ModelsPreprocessing:
    def __init__(self):
        self.system = "lin"
        self.deep_mlp_3_model = None
        self.deep_mlp_5_model = None
        self.deep_mlp_7_model = None
        self.autoencoder_model = None
        self.models_and_scaler_folder_path = self._select_path(
            win="D:/PWR/Praca magisterska/models/1_test",
            lin="/mnt/d/PWR/Praca magisterska/models/1_test"
        )
        self.meta_model_for_stacking = None
        self.scaler = None
        self.custom_objects = {
            "AutoencoderClassifier": AutoencoderClassifier,
            "DeepMLP_3": DeepMLP_3,
            "DeepMLP_5": DeepMLP_5,
            "DeepMLP_7": DeepMLP_7
        }
        self.load_models_and_scaler()

    def _select_path(self, win, lin):
        return win if self.system == "win" else lin

    def load_models_and_scaler(self):
        model_files = sorted(
            [f for f in os.listdir(self.models_and_scaler_folder_path) if f.endswith(".keras")]
        )
        meta_model_path = os.path.join(self.models_and_scaler_folder_path, "meta_model.pkl")
        scaler_files = [f for f in os.listdir(self.models_and_scaler_folder_path) if
                        f.endswith(".pkl") and "meta_model" not in f]

        self.load_models(model_files)
        self.load_meta_model(meta_model_path)
        self.load_scaler(scaler_files)
        print("Loading complete.")

    def load_models(self, model_files):
        for file_name in model_files:
            file_path = os.path.join(self.models_and_scaler_folder_path, file_name)
            if "AutoencoderClassifier" in file_name:
                self.autoencoder_model = load_model(file_path, custom_objects=self.custom_objects)
                print(f"Loaded Autoencoder model from: {file_name}")
            elif "_mlp_3_" in file_name or "DeepMLP_3" in file_name:
                self.deep_mlp_3_model = load_model(file_path, custom_objects=self.custom_objects)
                print(f"Loaded DeepMLP_3 model from: {file_name}")
            elif "_mlp_5_" in file_name or "DeepMLP_5" in file_name:
                self.deep_mlp_5_model = load_model(file_path, custom_objects=self.custom_objects)
                print(f"Loaded DeepMLP_5 model from: {file_name}")
            elif "_mlp_7_" in file_name or "DeepMLP_7" in file_name:
                self.deep_mlp_7_model = load_model(file_path, custom_objects=self.custom_objects)
                print(f"Loaded DeepMLP_7 model from: {file_name}")
            else:
                print(f"Unknown Keras model file: {file_name}")

    def load_meta_model(self, meta_model_path):
        if os.path.exists(meta_model_path):
            self.meta_model_for_stacking = joblib.load(meta_model_path)
            print(f"Loaded meta-model (stacking) from: meta_model.pkl")
        else:
            print(f"Meta-model file not found: {meta_model_path}")

    def load_scaler(self, scaler_files):
        if scaler_files:
            scaler_file_name = scaler_files[0]
            scaler_path = os.path.join(self.models_and_scaler_folder_path, scaler_file_name)
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from: {scaler_file_name}")
        else:
            print("No scaler file found.")





