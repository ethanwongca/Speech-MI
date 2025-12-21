"""
Step 4: Run Logistic Regression Probe (Train/Test Only)
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

NUM_NEGATIVE_SPEAKERS = 50 

def load_vectors(file_paths, layer_name):
    vectors = []
    for p in file_paths:
        try:
            data = torch.load(p, map_location="cpu")
            if layer_name in data:
                t = data[layer_name]
                if t.dim() == 3: t = t.squeeze(0)
                t_mean = t.mean(dim=0)
                vectors.append(t_mean.numpy())
        except Exception: 
            continue
    if not vectors: return None
    return np.stack(vectors)

def run_probe_for_layer(df, layer_name, timestep, target_speaker):
    subset = df[(df["block_name"] == layer_name) & (df["timestep"] == timestep)]

    # --- 1. TRAIN SET ---
    train_subset = subset[subset["split_mode"] == "train"]
    pos_train = train_subset[train_subset["speaker_id"] == target_speaker]["pt_file"].tolist()
    
    neg_train_df = train_subset[train_subset["speaker_id"] != target_speaker]
    all_neg_speakers = neg_train_df["speaker_id"].unique()
    if len(all_neg_speakers) == 0: return None
    
    chosen_neg = np.random.choice(all_neg_speakers, size=min(NUM_NEGATIVE_SPEAKERS, len(all_neg_speakers)), replace=False)
    neg_train = neg_train_df[neg_train_df["speaker_id"].isin(chosen_neg)]["pt_file"].tolist()

    if not pos_train or not neg_train: return None
    
    X_pos_train = load_vectors(pos_train, layer_name)
    X_neg_train = load_vectors(neg_train, layer_name)
    if X_pos_train is None or X_neg_train is None: return None
    
    X_train = np.concatenate([X_pos_train, X_neg_train])
    y_train = np.concatenate([np.ones(len(X_pos_train)), np.zeros(len(X_neg_train))])

    # --- 2. TEST SET ---
    test_subset = subset[subset["split_mode"] == "test"]
    pos_test = test_subset[test_subset["speaker_id"] == target_speaker]["pt_file"].tolist()
    neg_test = test_subset[test_subset["speaker_id"].isin(chosen_neg)]["pt_file"].tolist()
    
    X_test, y_test = None, None
    if pos_test and neg_test:
        X_pos_test = load_vectors(pos_test, layer_name)
        X_neg_test = load_vectors(neg_test, layer_name)
        if X_pos_test is not None and X_neg_test is not None:
            X_test = np.concatenate([X_pos_test, X_neg_test])
            y_test = np.concatenate([np.ones(len(X_pos_test)), np.zeros(len(X_neg_test))])

    # --- 3. EVAL SET (Optional) ---
    eval_subset = subset[subset["split_mode"] == "eval"]
    X_eval, y_eval = None, None
    
    if not eval_subset.empty:
        pos_eval = eval_subset[eval_subset["speaker_id"] == target_speaker]["pt_file"].tolist()
        neg_eval = eval_subset[eval_subset["speaker_id"].isin(chosen_neg)]["pt_file"].tolist()
        
        if pos_eval and neg_eval:
            X_pos_eval = load_vectors(pos_eval, layer_name)
            X_neg_eval = load_vectors(neg_eval, layer_name)
            if X_pos_eval is not None and X_neg_eval is not None:
                X_eval = np.concatenate([X_pos_eval, X_neg_eval])
                y_eval = np.concatenate([np.ones(len(X_pos_eval)), np.zeros(len(X_neg_eval))])

    # --- TRAIN & EVALUATE ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
    clf.fit(X_train_scaled, y_train)

    result = {
        "layer": layer_name,
        "timestep": timestep,
        "test_acc": None, "test_auc": None,
        "eval_acc": None, "eval_auc": None
    }

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        result["test_acc"] = accuracy_score(y_test, clf.predict(X_test_scaled))
        result["test_auc"] = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])

    if X_eval is not None:
        X_eval_scaled = scaler.transform(X_eval)
        result["eval_acc"] = accuracy_score(y_eval, clf.predict(X_eval_scaled))
        result["eval_auc"] = roc_auc_score(y_eval, clf.predict_proba(X_eval_scaled)[:, 1])
    
    return result

def main():
    if not Path(METADATA_PATH).exists(): 
        print(f"Error: {METADATA_PATH} not found.")
        return
        
    print(f"Loading metadata from {METADATA_PATH}...")
    df = pd.read_parquet(METADATA_PATH)
    
    train_counts = df[df["split_mode"]=="train"]["speaker_id"].value_counts()
    if train_counts.empty:
        print("Error: No training data found in metadata.")
        return
        
    target_speaker = str(train_counts.idxmax())
    print(f"Auto-selected Target Speaker: {target_speaker}")

    unique_layers = df["block_name"].unique().tolist()
    unique_timesteps = sorted(df["timestep"].unique().tolist())
    
    results = []
    print("Running probes...")
    for layer in tqdm(unique_layers, desc="Layers"):
        for t in unique_timesteps:
            res = run_probe_for_layer(df, layer, t, target_speaker)
            if res: results.append(res)

    results_df = pd.DataFrame(results)
    output_file = RESULTS_DIR / f"probe_results_speaker_{target_speaker}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()