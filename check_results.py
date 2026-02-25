import sys
from collections import defaultdict

def load_ground_truth(file_paths):
    """
    Carga los juicios de relevancia. Cuenta cuántos docs relevantes existen por query.
    """
    ground_truth = defaultdict(dict)
    relevance_counts = defaultdict(int)
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        query_id, _, doc_id, relevance = parts
                        is_relevant = int(relevance)
                        
                        # Solo guardamos si es relevante (1)
                        if is_relevant == 1:
                            ground_truth[query_id][doc_id] = 1
                            relevance_counts[query_id] += 1
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {file_path}", file=sys.stderr)
            sys.exit(1)
    return ground_truth, relevance_counts

def evaluate_results(results_file_path, ground_truth):
    query_hits = defaultdict(int)
    try:
        with open(results_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                # Formato esperado: query_id Q0 doc_id rank score ...
                query_id, doc_id = parts[0], parts[2]
                
                if query_id in ground_truth and doc_id in ground_truth[query_id]:
                    query_hits[query_id] += 1

    except FileNotFoundError:
        print(f"Error: No se encontró {results_file_path}", file=sys.stderr)
        sys.exit(1)
    return query_hits

def main():
    if len(sys.argv) != 2:
        print("Uso: python check_results.py <archivo_resultados.txt>", file=sys.stderr)
        sys.exit(1)

    results_file = sys.argv[1]
    
    # Asegúrate de que estas rutas sean correctas en tu sistema
    gt_files = [
        'archive/relevance_judgments_statutes.txt',
        'archive/relevance_judgments_priorcases.txt'
    ]

    print("Cargando Ground Truth...")
    ground_truth, total_relevant_per_query = load_ground_truth(gt_files)

    print(f"Evaluando {results_file}...")
    hits = evaluate_results(results_file, ground_truth)

    print("\n--- Resultados de Recall (Cobertura) ---")
    sorted_queries = sorted(total_relevant_per_query.keys())
    
    global_hits = 0
    global_total_relevant = 0

    for qid in sorted_queries:
        found = hits[qid]
        total = total_relevant_per_query[qid]
        
        global_hits += found
        global_total_relevant += total
        
        perc = (found / total * 100) if total > 0 else 0.0
        print(f"{qid}: Encontrados {found}/{total}. Recall: {perc:.2f}%")

    overall_recall = (global_hits / global_total_relevant * 100) if global_total_relevant > 0 else 0
    print("\n--- GLOBAL ---")
    print(f"Total Encontrados: {global_hits}/{global_total_relevant}")
    print(f"Recall Global: {overall_recall:.2f}%")

if __name__ == '__main__':
    main()