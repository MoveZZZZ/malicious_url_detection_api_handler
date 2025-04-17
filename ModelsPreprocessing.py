from CustomModels import DeepMLP_3, DeepMLP_5, DeepMLP_7, AutoencoderClassifier
from tensorflow.keras.models import load_model
import os
import joblib

class ModelsPreprocessing:
    def __init__(self):
        self.system = "lin"
        self.models_root_path = self._select_path(
            win="D:/PWR/Praca magisterska/models/1_test/AE_3_5_7_bert_smote_np_RFC_CF",
            lin="/mnt/d/PWR/Praca magisterska/models/1_test/AE_3_5_7_bert_smote_np_RFC_CF"
        )
        self.bert_models = []
        self.cf_models = []
        self.meta_model_for_stacking = None
        self.bert_scaler = None
        self.cf_scaler = None
        self.custom_objects = {
            "AutoencoderClassifier": AutoencoderClassifier,
            "DeepMLP_3": DeepMLP_3,
            "DeepMLP_5": DeepMLP_5,
            "DeepMLP_7": DeepMLP_7
        }


    def _select_path(self, win, lin):
        return win if self.system == "win" else lin

    def load_all_models_and_scaler(self):
        bert_path = os.path.join(self.models_root_path, "BERT_based")
        if os.path.exists(bert_path):
            self.bert_scaler = self.load_scaler_from_folder(bert_path, "BERT")
            self.bert_models = self.load_models_from_folder(bert_path)
        else:
            print(f"[WARN] BERT_models folder not found: {bert_path}")

        cf_path = os.path.join(self.models_root_path, "CF_based")
        if os.path.exists(cf_path):
            self.cf_scaler = self.load_scaler_from_folder(cf_path, "CF")
            self.cf_models = self.load_models_from_folder(cf_path)
        else:
            print(f"[WARN] CF_based folder not found: {cf_path}")

        meta_model_path = os.path.join(self.models_root_path, "meta_model.pkl")
        self.load_meta_model(meta_model_path)

        print("[INFO] Models and scalers successfully uploaded.")

    def load_models_from_folder(self, folder_path):
        loaded_models = []
        files = sorted(os.listdir(folder_path))

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            if file_name.startswith("minmax_scaler") and file_name.endswith(".pkl"):
                print(f"[SKIP] Skipping scaler file: {file_name}")
                continue

            try:
                if file_name.endswith(".keras"):
                    model = load_model(file_path, custom_objects=self.custom_objects)
                    loaded_models.append((file_name, model))
                    print(f"[OK] Loaded Keras model: {file_name}")

                elif file_name.endswith(".pkl"):
                    model = joblib.load(file_path)
                    loaded_models.append((file_name, model))
                    print(f"[OK] Loaded .pkl model: {file_name}")

                else:
                    print(f"[SKIP] Unsupported file type: {file_name}")

            except Exception as e:
                print(f"[ERROR] Failed to load model {file_name}: {e}")

        return loaded_models
    def load_scaler_from_folder(self, folder_path, model_type=""):
        try:
            scaler_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl") and f.startswith("minmax_scaler")]
            if not scaler_files:
                print(f"[WARN] No scaler found for {model_type} in folder: {folder_path}")
                return None
            scaler_file = scaler_files[0]
            scaler_path = os.path.join(folder_path, scaler_file)
            scaler = joblib.load(scaler_path)
            print(f"[OK] Loaded {model_type} scaler: {scaler_file}")
            return scaler
        except Exception as e:
            print(f"[ERROR] Failed to load scaler for {model_type}: {e}")
            return None

    def load_meta_model(self, meta_model_path):
        if os.path.exists(meta_model_path):
            try:
                self.meta_model_for_stacking = joblib.load(meta_model_path)
                print("[OK] Loaded meta-model (stacking).")
            except Exception as e:
                print(f"[ERROR] Failed to load meta-model: {e}")
        else:
            print("[WARN] Meta-model not found.")
