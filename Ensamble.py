from BertFeatureExtractor import BERTFeatureExtractor
from ModelsPreprocessing import ModelsPreprocessing
import numpy as np
from collections import Counter
from DataBaseLogic import init_db, DatabaseManager
class ModelEnsembler:
    def __init__(self):
        self._ModelsAndScaler = ModelsPreprocessing()
        self._BertFutureExtractor = BERTFeatureExtractor()
        self.label_mapping_url = {
            0: 'benign',
            1: 'defacement',
            2: 'phishing',
            3: 'malware'
        }
        init_db()
        self.db_manager = DatabaseManager()

    def extract_features_from_url(self, url):
        features = self._BertFutureExtractor.extract_features_bert([url])
        features_scaled = self._ModelsAndScaler.scaler.transform(features)
        features_array = np.array(features_scaled)
        return features_array

    def get_prediction_list(self, extracted_features):
        pred_list = []
        if self._ModelsAndScaler.deep_mlp_3_model is not None:
            preds_3 = self._ModelsAndScaler.deep_mlp_3_model.predict(extracted_features)
            pred_list.append(preds_3)
        if self._ModelsAndScaler.deep_mlp_5_model is not None:
            preds_5 = self._ModelsAndScaler.deep_mlp_5_model.predict(extracted_features)
            pred_list.append(preds_5)
        if self._ModelsAndScaler.deep_mlp_7_model is not None:
            preds_7 = self._ModelsAndScaler.deep_mlp_7_model.predict(extracted_features)
            pred_list.append(preds_7)
        if self._ModelsAndScaler.autoencoder_model is not None:
            preds_ae = self._ModelsAndScaler.autoencoder_model.predict(extracted_features)["classifier"]
            pred_list.append(preds_ae)
        if len(pred_list) == 0:
            raise ValueError("No base models found to predict.")
        predictions_stacked = np.hstack(pred_list)
        return predictions_stacked

    def majority_vote_prediction(self, extracted_features):
        votes = []
        vote_probs = []

        if self._ModelsAndScaler.deep_mlp_3_model is not None:
            preds = self._ModelsAndScaler.deep_mlp_3_model.predict(extracted_features)
            label = int(np.argmax(preds, axis=1)[0])
            prob = float(np.max(preds, axis=1)[0])
            votes.append(label)
            vote_probs.append(prob)

        if self._ModelsAndScaler.deep_mlp_5_model is not None:
            preds = self._ModelsAndScaler.deep_mlp_5_model.predict(extracted_features)
            label = int(np.argmax(preds, axis=1)[0])
            prob = float(np.max(preds, axis=1)[0])
            votes.append(label)
            vote_probs.append(prob)

        if self._ModelsAndScaler.deep_mlp_7_model is not None:
            preds = self._ModelsAndScaler.deep_mlp_7_model.predict(extracted_features)
            label = int(np.argmax(preds, axis=1)[0])
            prob = float(np.max(preds, axis=1)[0])
            votes.append(label)
            vote_probs.append(prob)

        if self._ModelsAndScaler.autoencoder_model is not None:
            preds = self._ModelsAndScaler.autoencoder_model.predict(extracted_features)["classifier"]
            label = int(np.argmax(preds, axis=1)[0])
            prob = float(np.max(preds, axis=1)[0])
            votes.append(label)
            vote_probs.append(prob)

        if not votes:
            raise ValueError("No base models found for majority vote prediction.")

        vote_counts = Counter(votes)
        majority_label, count = vote_counts.most_common(1)[0]
        if count > len(votes) / 2:
            percentage_votes = round((count / len(votes)) * 100, 2)
            majority_probs = [vote_probs[i] for i, v in enumerate(votes) if v == majority_label]
            avg_probability = sum(majority_probs) / len(majority_probs)
            avg_percentage_confidence = round(avg_probability * 100, 2)
            final_label = self.label_mapping_url.get(majority_label, "unknown")
            return final_label, percentage_votes, avg_percentage_confidence
        else:
            return None

    def stacking_vote_info(self, extracted_features, final_label_id):
        votes = []
        confidences = []

        if self._ModelsAndScaler.deep_mlp_3_model is not None:
            preds = self._ModelsAndScaler.deep_mlp_3_model.predict(extracted_features)
            label = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds, axis=1)[0])
            votes.append(label)
            confidences.append(confidence)

        if self._ModelsAndScaler.deep_mlp_5_model is not None:
            preds = self._ModelsAndScaler.deep_mlp_5_model.predict(extracted_features)
            label = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds, axis=1)[0])
            votes.append(label)
            confidences.append(confidence)

        if self._ModelsAndScaler.autoencoder_model is not None:
            preds = self._ModelsAndScaler.autoencoder_model.predict(extracted_features)["classifier"]
            label = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds, axis=1)[0])
            votes.append(label)
            confidences.append(confidence)
        total_models = len(votes)
        if total_models == 0:
            raise ValueError("No base models found for stacking vote info.")
        matching_votes = [votes[i] for i in range(total_models) if votes[i] == final_label_id]
        vote_percentage = round((len(matching_votes) / total_models) * 100, 2)
        matching_confidences = [confidences[i] for i in range(total_models) if votes[i] == final_label_id]
        if matching_confidences:
            avg_confidence = round((sum(matching_confidences) / len(matching_confidences)) * 100, 2)
        else:
            avg_confidence = 0.0

        return vote_percentage, avg_confidence

    def predict_url(self, url: str):
        extracted_features = self.extract_features_from_url(url)

        majority_result = self.majority_vote_prediction(extracted_features)
        if majority_result is not None:
            final_label, vote_percentage, avg_confidence = majority_result
            meta_confidence = None  # В ветке majority vote meta confidence отсутствует
            print(f"Majority vote: {final_label} ({vote_percentage}% votes, medium confidence: {avg_confidence}%)")
        else:
            base_preds = self.get_prediction_list(extracted_features)
            if self._ModelsAndScaler.meta_model_for_stacking is None:
                raise ValueError("Meta-model for stacking is not loaded.")
            final_pred = self._ModelsAndScaler.meta_model_for_stacking.predict(base_preds)
            final_proba = self._ModelsAndScaler.meta_model_for_stacking.predict_proba(base_preds)
            final_label_id = int(final_pred[0])
            final_label = self.label_mapping_url.get(final_label_id, "unknown")
            max_probability_for_class = final_proba[0][final_label_id]
            stacking_confidence = round(max_probability_for_class * 100, 2)
            vote_percentage, avg_confidence = self.stacking_vote_info(extracted_features, final_label_id)
            meta_confidence = stacking_confidence
            print(f"Stacking: {final_label} (meta confidence: {stacking_confidence}%, vote percentage: {vote_percentage}%, avg base confidence: {avg_confidence}%)")
        if self.db_manager.get_scan_result(url) is None:
            self.db_manager.add_scan_result(url, vote_percentage, avg_confidence, meta_confidence, final_label)
            print(f"Запись для {url} успешно добавлена в базу.")
        else:
            print(f"Запись для {url} уже существует в базе.")


        return final_label, avg_confidence