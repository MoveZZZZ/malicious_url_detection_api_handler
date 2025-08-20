import time
import threading
from tqdm import tqdm
import pandas as pd
import requests
import zipfile
from io import BytesIO
import random
import platform
from Ensamble import ModelEnsembler
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import sys
from tabulate import tabulate


import os
import builtins
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
_ORIGINAL_PRINT = builtins.print

def disable_print():
    builtins.print = lambda *args, **kwargs: None

def enable_print():
    builtins.print = _ORIGINAL_PRINT

@contextmanager
def redirect_stdout_only(file_path):
    original_stdout = sys.stdout
    with open(file_path, "w", encoding="utf-8") as f:
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = original_stdout

def download_malware_urls(limit=10000):
    print("Downloading malware links from URLHaus")
    url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
    response = requests.get(url)
    columns = [
        "id", "dateadded", "url", "url_status", "last_online", "threat",
        "tags", "urlhaus_link", "reporter"
    ]
    df = pd.read_csv(BytesIO(response.content), sep=",", comment="#", names=columns, skiprows=1)
    df = df[df['url_status'] == 'online']
    urls = df['url'].dropna().unique().tolist()
    random.shuffle(urls)
    return urls[:limit]

def download_phishing_urls(limit=10000):
    print("Downloading phishing links from OpenPhish")
    phish_final = []

    response = requests.get("https://openphish.com/feed.txt")
    if response.status_code == 200:
        openphish_urls = list(set(response.text.strip().split('\n')))
        random.shuffle(openphish_urls)
        phish_final.extend(openphish_urls)
        print(f"From openphish: {len(openphish_urls)}")

    if len(phish_final) < limit:
        print("Adding URLs from Phishing.Database...")
        backup_url = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links.txt"
        response = requests.get(backup_url)
        if response.status_code == 200:
            extra_urls = list(set(response.text.strip().split('\n')))
            extra_urls = [url for url in extra_urls if url and not url.startswith("#")]
            random.shuffle(extra_urls)
            to_add = limit - len(phish_final)
            phish_final.extend(extra_urls[:to_add])
            print(f"Added from second database : {min(to_add, len(extra_urls))}")

    return list(set(phish_final))[:limit]

def download_benign_urls(limit=10000):
    print("Downloading benign links from Tranco")
    safe_url = "https://tranco-list.eu/top-1m.csv.zip"
    response = requests.get(safe_url)
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        with z.open(z.namelist()[0]) as f:
            safe_df = pd.read_csv(f, names=["rank", "domain"])
    safe_urls = ["https://" + d for d in safe_df["domain"].dropna().unique().tolist()]
    random.shuffle(safe_urls)
    return safe_urls[:limit]

def check_duplicates(new_df):
    existing_dataset_path = "F:/Datasets/CLEARED_BASE_DATASET/dataset_mal_cleared_full.csv" \
        if platform.system() == "Windows" \
        else "/mnt/f/Datasets/CLEARED_BASE_DATASET/dataset_mal_cleared_full.csv"
    if os.path.exists(existing_dataset_path):
        print(f"\nChecking repeated samples : {existing_dataset_path}")
        existing_df = pd.read_csv(existing_dataset_path)
        existing_urls = set(existing_df['url'].dropna().unique().tolist())
        new_urls = set(new_df['url'])
        duplicates = existing_urls.intersection(new_urls)
        print(f"Number of repeated samples: {len(duplicates)}")
        if duplicates:
            print("Repeated sample example:")
            for url in list(duplicates)[:10]:
                print(" -", url)
    else:
        print(f"Base file {existing_dataset_path} not found")

def check_url_set(df):
    predictions = []
    _Ensambler = ModelEnsembler()
    start_time = time.perf_counter()

    for url in tqdm(df['url'], desc="Checking URLs", ncols=100):
        try:
            label, _ = _Ensambler.predict_url(url)
        except Exception as e:
            print(f"[ERROR] {url} failed: {e}")
            label = "error"
        predictions.append(label)

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    urls_per_sec = len(df) / elapsed
    enable_print()
    print(f"\nClassification completed in {minutes}m {seconds}s ({elapsed:.2f} sec)")
    print(f"Speed: {urls_per_sec:.2f} URL/сек")

    df = df.copy()
    df['predicted'] = predictions
    return df


def check_url_set_parallel(df, num_threads=4):
    df_chunks = np.array_split(df, num_threads)
    progress_bar = tqdm(total=len(df), desc="Checking URLs", ncols=100)
    tqdm_lock = threading.Lock()
    tqdm.set_lock(tqdm_lock)
    prediction_times = []
    def worker(df_chunk):
        predictions = []
        ensambler = ModelEnsembler()
        start = time.perf_counter()
        for url in df_chunk['url']:
            try:
                label, _ = ensambler.predict_url(url)
            except Exception as e:
                print(f"[ERROR] {url} failed: {e}")
                label = "error"
            predictions.append(label)
            progress_bar.update(1)
        end = time.perf_counter()
        prediction_times.append(end - start)
        df_chunk = df_chunk.copy()
        df_chunk['predicted'] = predictions
        return df_chunk

    result_dfs = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, chunk) for chunk in df_chunks]
        for future in as_completed(futures):
            result_dfs.append(future.result())

    progress_bar.close()
    total_classification_time = sum(prediction_times)
    urls_per_sec = len(df) / total_classification_time
    minutes = int(total_classification_time // 60)
    seconds = int(total_classification_time % 60)
    enable_print()
    print(f"\nNet classification time: {minutes}m {seconds}s ({total_classification_time:.2f} sec).")
    print(f"Speed: {urls_per_sec:.2f} URL/s")

    final_df = pd.concat(result_dfs, ignore_index=True)
    return final_df
def classification_report(df_with_predictions):
    total = len(df_with_predictions)
    correct = sum(df_with_predictions['label'] == df_with_predictions['predicted'])
    incorrect = total - correct

    critical_fp = df_with_predictions[
        (df_with_predictions['label'].isin(['malware', 'phishing'])) &
        (df_with_predictions['predicted'] == 'benign')
    ]
    non_critical_fp = incorrect - len(critical_fp)

    print("\nClassification Report\n")


    metrics = [
        ["Total URLs", total, "100.00%"],
        ["Correct classifications", correct, f"{(correct / total * 100):.2f}%"],
        ["Incorrect classifications", incorrect, f"{(incorrect / total * 100):.2f}%"],
        ["Critical false positives\n(malware/phishing → benign)", len(critical_fp), f"{(len(critical_fp)/total*100):.2f}%"],
        ["Non-critical false positives\n(benign → other)", non_critical_fp, f"{(non_critical_fp/total*100):.2f}%"]
    ]
    print(tabulate(metrics, headers=["Metric", "Count", "Percent"], tablefmt="grid"))


    print("\nURL distribution by category:\n")
    label_counts = df_with_predictions['label'].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    print(tabulate(label_counts, headers="keys", tablefmt="fancy_grid"))

    mismatches = df_with_predictions[df_with_predictions['label'] != df_with_predictions['predicted']]
    if not mismatches.empty:
        print("\nMismatched examples:\n")
        print(tabulate(mismatches[['url', 'label', 'predicted']].head(5), headers="keys", tablefmt="github"))
    else:
        print("\nNo mismatched examples found.")

def check_file_exist_and_get_dataset():
    file_path = "test_urls.csv"
    if os.path.exists(file_path):
        print("File found, loading from CSV...")
        df = pd.read_csv(file_path)
    else:
        malware = download_malware_urls()
        phishing = download_phishing_urls()
        benign = download_benign_urls()
        df = pd.DataFrame({
            "url": malware + phishing + benign,
            "label": ["malware"] * len(malware) + ["phishing"] * len(phishing) + ["benign"] * len(benign)
        }).sample(frac=1, random_state=42).reset_index(drop=True)

        df.to_csv(file_path, index=False)
    return df

if __name__ == "__main__":
    log_file = os.path.join("/mnt/d/PWR/Praca magisterska/models/1_test/AE_3_5_7_bert_smote_np_RFC_CF", "output.txt")
    with redirect_stdout_only(log_file):
        df = check_file_exist_and_get_dataset()
        disable_print()
        df_result = check_url_set_parallel(df[0:10], num_threads=6)
        classification_report(df_result)
        check_duplicates(df_result)







