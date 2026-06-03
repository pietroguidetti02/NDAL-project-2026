import os
import subprocess
import shutil
import time

# I tre esperimenti principali che vogliamo lanciare in parallelo
configs = [
    "config/exp2.yaml",              
    "config/exp3.yaml", 
    "config/exp_time_split.yaml",    # Train su First Window, Test su Second Window
    "config/exp_spatial_split.yaml"  # Train su A->B/A->C, Test sui link di C
]

processes = []
temp_files = []

print("=== Inizio Lancio Esperimenti in Parallelo ===")

for i, config_path in enumerate(configs):
    if not os.path.exists(config_path):
        print(f"File {config_path} non trovato. Lo salto.")
        continue
        
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Creazione copia sicura temporanea dello script
    temp_script_name = f"temp_main_LSTM_{config_name}.py"
    shutil.copy2("main_comparison_LSTM.py", temp_script_name)
    temp_files.append(temp_script_name)
    
    # Creazione file di log per non intasare la console con stampe miste
    log_file_name = f"results/log_{config_name}.txt"
    os.makedirs("results", exist_ok=True)
    log_file = open(log_file_name, "w")
    
    print(f"[*] Lancio l'esperimento {config_name} in background. Log -> {log_file_name}")
    
    # Esecuzione asincrona (Popen)
    p = subprocess.Popen(["python", temp_script_name, "--config", config_path], stdout=log_file, stderr=subprocess.STDOUT)
    processes.append((config_name, p, log_file))
    
    # Pausa di 2 secondi per evitare micro-collisioni sui timestamp
    time.sleep(2)

print("\nTutti gli esperimenti sono stati lanciati. Attendo la loro conclusione (potrebbe volerci un po')...")

# Aspettiamo che finiscano tutti
for config_name, p, log_file in processes:
    p.wait()
    log_file.close()
    
    if p.returncode == 0:
        print(f"[OK] Esperimento {config_name} terminato con successo!")
    else:
        print(f"[ERRORE] Esperimento {config_name} terminato con errori. Controlla il log.")

print("\nPulizia delle copie temporanee sicure in corso...")
for temp_file in temp_files:
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
print("=== Finito! Controlla la cartella 'results' per i grafici e i log. ===")
