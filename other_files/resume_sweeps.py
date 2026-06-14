import argparse
import os
import sys
import yaml
from joblib import Parallel, delayed

# Assicuriamoci che i path siano corretti per l'import del progetto
sys.path.append(os.getcwd())

from src.data_loader import load_and_split_data
from main_sweep import run_single_experiment

def get_config_from_log(target_dir):
    """Cerca di dedurre il file di configurazione usato dal log originale."""
    log_path = os.path.join(target_dir, "sweep_log.txt")
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "[*] Command:" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("--config")
                        return parts[idx + 1]
                    except ValueError:
                        pass
    except Exception:
        pass
    return None

def is_experiment_completed(exp_dir, N, X):
    """Controlla se una data combinazione (N, X) è stata completata con successo."""
    log_file = os.path.join(exp_dir, f"log_N_{N}_X_{X}.txt")
    if not os.path.exists(log_file):
        return False
    try:
        with open(log_file, "r") as f:
            content = f.read()
            # Il tag finale che viene stampato esclusivamente alla fine di run_single_experiment
            if f"[Run N={N}, X={X}] Completed." in content:
                return True
    except Exception:
        pass
    return False

def process_target_dir(target_dir, max_cores):
    print(f"\n{'='*60}")
    print(f"[*] Analisi della cartella: {target_dir}")
    
    config_path = get_config_from_log(target_dir)
    if not config_path or not os.path.exists(config_path):
        print(f"[!] File config non deducibile dal log per {target_dir}.")
        # Se la cartella contiene info nel nome, usiamo un fallback intelligente
        if "time_split" in target_dir:
            config_path = 'config/sweep_time_split.yaml'
        elif "spatial_split" in target_dir:
            config_path = 'config/sweep_spatial_split.yaml'
        else:
            config_path = 'config/sweep_time_split.yaml'
        print(f"[*] Utilizzo config di fallback: {config_path}")
    else:
        print(f"[*] File config dedotto con successo: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    N_values = config.get('N_values', [10, 15, 20, 30, 60])
    X_values = config.get('X_values', [1, 5, 10, 15, 20])
    
    all_combinations = [(n, x) for n in N_values for x in X_values if x < n]
    missing_combinations = []
    
    for n, x in all_combinations:
        exp_dir = os.path.join(target_dir, f"N_{n}_X_{x}")
        if is_experiment_completed(exp_dir, n, x):
            print(f"  -> [N={n}, X={x}]: COMPLETATO")
        else:
            print(f"  -> [N={n}, X={x}]: MANCANTE o INTERROTTO")
            missing_combinations.append((n, x))
            
    if not missing_combinations:
        print(f"[*] Tutti gli esperimenti in {target_dir} sono già completati!")
        return
        
    print(f"\n[*] Trovati {len(missing_combinations)} esperimenti da eseguire/riprendere in {target_dir}.")
    print("[*] Caricamento del dataset in RAM (fatto una sola volta per i worker)...")
    train_dfs_dict, test_dfs_dict = load_and_split_data(config)
    
    print(f"[*] Avvio simulazioni mancanti con max_cores={max_cores}...")
    Parallel(n_jobs=max_cores)(
        delayed(run_single_experiment)(n, x, config, target_dir, train_dfs_dict, test_dfs_dict)
        for n, x in missing_combinations
    )
    
    print(f"\n[*] Ripristino completato per la cartella: {target_dir}")

def main():
    parser = argparse.ArgumentParser(description='Ripristina esperimenti sweep interrotti causa OOM.')
    parser.add_argument('target_dirs', nargs='+', help='Cartelle target degli esperimenti (es. results/sweep_2026...)')
    parser.add_argument('--max_cores', type=int, default=3, help='Numero massimo di processi (consigliato 2-4 per non saturare la RAM)')
    args = parser.parse_args()
    
    for t_dir in args.target_dirs:
        if not os.path.isdir(t_dir):
            print(f"[!] Attenzione: La directory '{t_dir}' non esiste. Salto...")
            continue
        process_target_dir(t_dir, args.max_cores)

if __name__ == '__main__':
    main()
