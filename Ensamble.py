from BertFeatureExtractor import BERTFeatureExtractor
from ModelsPreprocessing import ModelsPreprocessing
import numpy as np
from collections import Counter
from DataBaseLogic import init_db, DatabaseManager
from CustomFeaturesExrtactor import CustomFeatureExtractor
class ModelEnsembler:
    def __init__(self):
        self._ModelsAndScaler = ModelsPreprocessing()
        self._BertFutureExtractor = BERTFeatureExtractor()
        self._CustomFeatureExtractor = CustomFeatureExtractor()
        self.label_mapping_url = {
            0: 'benign',
            1: 'defacement',
            2: 'phishing',
            3: 'malware'
        }
        init_db()
        self.db_manager = DatabaseManager()
        self._ModelsAndScaler.load_all_models_and_scaler()

    def extract_features_from_url(self, url):
        features = self._BertFutureExtractor.extract_features_bert([url])
        features_scaled = self._ModelsAndScaler.bert_scaler.transform(features)
        return np.array(features_scaled)

    def extract_custom_features_from_url(self, url):
        features = self._CustomFeatureExtractor.extract_features(url)
        features_scaled = self._ModelsAndScaler.cf_scaler.transform(features)
        return np.array(features_scaled)

    def predict_with_optional_verbose(self,model, features):
        try:
            return model.predict(features, verbose=0)
        except TypeError:
            return model.predict(features)

    def check_and_change_prediction_dim(self, preds):
        if preds.ndim == 1 or preds.shape[1] == 1:
            num_classes = len(self.label_mapping_url)
            preds_oh = np.zeros((preds.shape[0], num_classes))
            preds_oh[np.arange(preds.shape[0]), preds.astype(int)] = 1
            preds = preds_oh

        return preds
    def predict_with_model(self, model_obj, features):
        model_name, model = model_obj
        try:
            preds = self.predict_with_optional_verbose(model,features)
            if isinstance(preds, dict):
                if "classifier" in preds:
                    preds = preds["classifier"]
                else:
                    raise ValueError(f"[ERROR] Model {model_name} returned dict but no 'classifier' key found.")
            return self.check_and_change_prediction_dim(preds)
        except Exception as e:
            print(f"[ERROR] Predict failed for model {model_name}: {e}")
            return None

    def extract_all_features(self, url):
        cf_features = None
        bert_features = None
        if self._ModelsAndScaler.cf_models is not None:
            cf_features = self.extract_custom_features_from_url(url)
        if self._ModelsAndScaler.bert_models is not None:
            bert_features = self.extract_features_from_url(url)
        return cf_features, bert_features


    def get_prediction_list(self, url):
        pred_list = []
        cf_features, bert_features = self.extract_all_features(url)

        for model in self._ModelsAndScaler.bert_models:
            preds = self.predict_with_model(model, bert_features)
            if preds is not None:
                pred_list.append(preds)

        for model in self._ModelsAndScaler.cf_models:
            preds = self.predict_with_model(model, cf_features)
            if preds is not None:
                pred_list.append(preds)

        if not pred_list:
            raise ValueError("No base models returned valid predictions.")
        return np.hstack(pred_list)

    def calculate_votes_and_final_label(self, votes, vote_probs):
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

    def majority_vote_prediction(self, url):
        votes = []
        vote_probs = []
        cf_features, bert_features = self.extract_all_features(url)

        for model in self._ModelsAndScaler.bert_models:
            preds = self.predict_with_model(model, bert_features)
            if preds is not None:
                label = int(np.argmax(preds, axis=1)[0])
                prob = float(np.max(preds, axis=1)[0])
                votes.append(label)
                vote_probs.append(prob)

        for model in self._ModelsAndScaler.cf_models:
            preds = self.predict_with_model(model, cf_features)
            if preds is not None:
                label = int(np.argmax(preds, axis=1)[0])
                prob = float(np.max(preds, axis=1)[0])
                votes.append(label)
                vote_probs.append(prob)

        if not votes:
            raise ValueError("No base models found for majority vote.")

        return self.calculate_votes_and_final_label(votes, vote_probs)
    def calculate_stacking_result(self, votes, final_label_id, confidences):
        total_models = len(votes)
        if total_models == 0:
            raise ValueError("No base models found for stacking vote info.")

        matching_votes = [votes[i] for i in range(total_models) if votes[i] == final_label_id]
        vote_percentage = round((len(matching_votes) / total_models) * 100, 2)
        matching_confidences = [confidences[i] for i in range(total_models) if votes[i] == final_label_id]
        avg_confidence = round((sum(matching_confidences) / len(matching_confidences)) * 100,
                               2) if matching_confidences else 0.0

        return vote_percentage, avg_confidence
    def stacking_vote_info(self, url, final_label_id):
        votes = []
        confidences = []

        bert_features = self.extract_features_from_url(url)
        cf_features = self.extract_custom_features_from_url(url)

        for model in self._ModelsAndScaler.bert_models:
            preds = self.predict_with_model(model, bert_features)
            if preds is not None:
                label = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds, axis=1)[0])
                votes.append(label)
                confidences.append(confidence)

        for model in self._ModelsAndScaler.cf_models:
            preds = self.predict_with_model(model, cf_features)
            if preds is not None:
                label = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds, axis=1)[0])
                votes.append(label)
                confidences.append(confidence)
        return self.calculate_stacking_result(votes,final_label_id,confidences)

    def predict_stacking(self, url):
        base_preds = self.get_prediction_list(url)
        if self._ModelsAndScaler.meta_model_for_stacking is None:
            raise ValueError("Meta-model for stacking is not loaded.")
        final_pred = self._ModelsAndScaler.meta_model_for_stacking.predict(base_preds)
        final_proba = self._ModelsAndScaler.meta_model_for_stacking.predict_proba(base_preds)
        final_label_id = int(final_pred[0])
        final_label = self.label_mapping_url.get(final_label_id, "unknown")
        max_probability_for_class = final_proba[0][final_label_id]
        stacking_confidence = round(max_probability_for_class * 100, 2)
        vote_percentage, avg_confidence = self.stacking_vote_info(url, final_label_id)
        return final_label, stacking_confidence, vote_percentage, avg_confidence

    def write_results(self, url, vote_percentage, avg_confidence, meta_confidence, final_label):
        if self.db_manager.get_scan_result(url) is None:
            self.db_manager.add_scan_result(url, vote_percentage, avg_confidence, meta_confidence, final_label)
            print(f"Record for successfully added.")
        else:
            print(f"Record for already exists.")

    def predict_url(self, url: str):
        majority_result = self.majority_vote_prediction(url)
        if majority_result is not None:
            final_label, vote_percentage, avg_confidence = majority_result
            meta_confidence = None
            print(f"Majority vote: {final_label} ({vote_percentage}% votes, medium confidence: {avg_confidence}%)")
        else:
            final_label, meta_confidence, vote_percentage, avg_confidence = self.predict_stacking(url)
            print(f"Stacking: {final_label} (meta confidence: {meta_confidence}%, vote percentage: {vote_percentage}%, avg base confidence: {avg_confidence}%)")

        self.write_results(url, vote_percentage, avg_confidence, meta_confidence, final_label)

        return final_label, avg_confidence
