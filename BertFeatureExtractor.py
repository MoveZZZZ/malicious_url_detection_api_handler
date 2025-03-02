from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import pandas as pd

class BERTFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
    def extract_features_bert(self, texts, batch_size=1):
        all_features = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            encoded_dict = self.tokenizer.batch_encode_plus(
                batch_texts, add_special_tokens=True, padding="longest",
                truncation=True, max_length=128, return_tensors="pt"
            )
            input_ids = encoded_dict["input_ids"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                hidden_states = outputs.hidden_states
            token_vecs = torch.mean(torch.stack(hidden_states[-4:]), dim=0)
            batch_features = torch.mean(token_vecs, dim=1)
            all_features.append(batch_features.cpu())
        features_array = torch.cat(all_features, dim=0).numpy()
        num_features = features_array.shape[1]
        feature_names = [f"feature_{i}" for i in range(num_features)]
        df_features = pd.DataFrame(features_array, columns=feature_names)
        return df_features