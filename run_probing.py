"""
Step 4: Run Logistic Regression Probe (Pandas Version)
"""
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

METADATA_PATH = "./extracted_features_fixed_text/probing_metadata.parquet"
RESULTS_DIR = Path("./probing_results")
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_SPEAKER = "100" 
NUM_NEGATIVE_SPEAKERS = 50 

def load_vectors(file_paths):
    vectors = []
    for p in file_paths:
        try:
            t = torch.load(p, map_location="cpu")
            if t.dim() == 3: t = t.squeeze(0)
            t_mean = t.mean(dim=0)
            vectors.append(t_mean.numpy())
        except Exception: continue
    if not vectors: return None
    return np.stack(vectors)

def run_probe_for_layer(df, layer_name, timestep):
    # 1. Filter Data
    subset = df[(df["block_name"] == layer_name) & (df["timestep"] == timestep)]

    # 2. Train Set
    train_subset = subset[subset["split_mode"] == "train"]
    pos_train_paths = train_subset[train_subset["speaker_id"] == TARGET_SPEAKER]["pt_file"].tolist()
    
    neg_train_df = train_subset[train_subset["speaker_id"] != TARGET_SPEAKER]
    all_neg_speakers = neg_train_df["speaker_id"].unique()
    if len(all_neg_speakers) == 0: return None
    
    chosen_neg_speakers = np.random.choice(
        all_neg_speakers, 
        size=min(NUM_NEGATIVE_SPEAKERS, len(all_neg_speakers)), 
        replace=False
    )
    neg_train_paths = neg_train_df[neg_train_df["speaker_id"].isin(chosen_neg_speakers)]["pt_file"].tolist()

    if not pos_train_paths or not neg_train_paths: return None
    X_pos_train = load_vectors(pos_train_paths)
    X_neg_train = load_vectors(neg_train_paths)
    if X_pos_train is None or X_neg_train is None: return None
    
    X_train = np.concatenate([X_pos_train, X_neg_train])
    y_train = np.concatenate([np.ones(len(X_pos_train)), np.zeros(len(X_neg_train))])

    # 3. Test Set
    test_subset = subset[subset["split_mode"] == "test"]
    pos_test_paths = test_subset[test_subset["speaker_id"] == TARGET_SPEAKER]["pt_file"].tolist()
    neg_test_paths = test_subset[test_subset["speaker_id"].isin(chosen_neg_speakers)]["pt_file"].tolist()

    if not pos_test_paths or not neg_test_paths: return None
    X_pos_test = load_vectors(pos_test_paths)
    X_neg_test = load_vectors(neg_test_paths)
    if X_pos_test is None or X_neg_test is None: return None

    X_test = np.concatenate([X_pos_test, X_neg_test])
    y_test = np.concatenate([np.ones(len(X_pos_test)), np.zeros(len(X_neg_test))])

    # 4. Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    try: X_test_scaled = scaler.transform(X_test)
    except ValueError: return None

    clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
    clf.fit(X_train_scaled, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    return {
        "layer": layer_name,
        "timestep": timestep,
        "test_acc": accuracy_score(y_test, y_pred),
        "test_auc": roc_auc_score(y_test, y_prob)
    }

def main():
    print(f"Loading metadata from {METADATA_PATH}...")
    df = pd.read_parquet(METADATA_PATH)
    
    global TARGET_SPEAKER
    if TARGET_SPEAKER not in df["speaker_id"].unique():
        TARGET_SPEAKER = df["speaker_id"].unique()[0]
        print(f"Target '100' not found. Switching to {TARGET_SPEAKER}")

    unique_layers = df["block_name"].unique().tolist()
    unique_timesteps = sorted(df["timestep"].unique().tolist())
    
    print(f"Target: {TARGET_SPEAKER} | Layers: {len(unique_layers)} | Steps: {len(unique_timesteps)}")
    
    results = []
    for layer in tqdm(unique_layers, desc="Layers"):
        for t in unique_timesteps:
            res = run_probe_for_layer(df, layer, t)
            if res: results.append(res)

    results_df = pd.DataFrame(results)
    save_path = RESULTS_DIR / f"probe_results_speaker_{TARGET_SPEAKER}.csv"
    results_df.to_csv(save_path, index=False)
    
    print("\nTop 5 Layers for Speaker Identity:")
    if not results_df.empty:
        print(results_df.sort_values("test_auc", ascending=False).head(5))

if __name__ == "__main__":
    main()