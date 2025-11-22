import os, json
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

# Configuración
MODEL_NAME = "BAAI/bge-base-en-v1.5"  # cambiar a otro BGE si quieres
NORMALIZE = True                     # True para COSINE
CASEDOC_DIR = "../archive/Object_casedocs"
STATUTE_DIR = "../archive/Object_statutes"
QUERIES = [
    "Represent this sentence for searching relevant passages: The appellant before us was examined as prime witness in the trial of T.R. on the file of the Special Judge against the first respondent. The trial ended in conviction against the first respondent and when the appeal filed by him came to be heard by the High Court the appellant had become a Cabinet Minister. On account of the disparaging remarks made by the Appellate Judge the appellant tendered his resignation and demitted office for maintaining democratic traditions. It is in that backgroud this appeal has come to be preferred. Pursuant to a trap laid by the Vigilance Police on the complaint of the appellant's Manager, P1 (P.W.2) the first respondent was arrested on 26.4.79 for having accepted a bribe of Rs. 2,000 from P1. The marked currency notes were recovered from the brief case of the first respondent prior to the arrest. The prosecution case was that the first respondent had been extracting illegal gratification at the rate of Rs. 1,000 er month during the months of January, February and March, 1979 from P1 but all of a sudden he raised the demand to Rs. 2,000 per month in April 1979 and this led to P1 laying information (Exhibit I) before the Superintendent of Police (Vigilance). Acting on the report, a trap was laid on 26.4.79 and after P1 had handed over the marked currency notes the Vigilance party entered the office and recovered the currency notes from the brief case and arrested the first respondent. The first respondent denied having received any illegal gratification but offered no explanation for the presence of the currency notes in his brief case. Eleven witnesses including the appellant who figured as P.W.8 were examined by the prosecution and the first respondent examined three witnesses D.Ws. 1 to 3 to substantiate the defence set up by him, viz., that the sum of Rs. 2,000 had been paid by way of donation for conducting a drama and publishing a souvenir by the Mining Officers' Club and also towards donation for Children's Welfare Fund.The Special Judge accepted the prosecution case and held the first respondent guilty. The Special Judge awarded a sentence of rigorous imprisonment for one year for the conviction under the first charge but did not award any separate sentence for the conviction under the second. Against the conviction and sentence the first respondent appealed to the High Court. A learned Judge of the High Court has allowed the appeal holding that the prosecution has not proved its case by acceptable evidence and besides, the first respondent's explanation for the possession of the currency notes appeared probable. While acquitting the first respondent the learned Judge has, however, made several adverse remarks about the conduct of the appellant and about the credibility of his testimony and it is with that part of the judgment we are now concerned with in this appeal.",
    "Represent this sentence for searching relevant passages: This appeal is preferred against the judgment dated 19.8.2011 passed by the High Court, whereby the High Court partly allowed the appeal filed by the appellants thereby confirming the conviction of the appellants with certain modifications. On 18.11.1994, at about 8.00 A.M. in the morning the complainant P1 (PW-5) along with his two sons namely P2 and P3 (PW-6) were busy in cutting pullas (reeds) from the dola of their field. At that time, P4 (A-1) and his sons P5 (A-2), P6 (A-3) and P7 (A-4) armed with jaily, pharsi and lathis respectively, entered the land where the complainant was working with his sons and asked them not to cut the pullas as it was jointly held by both the parties. Wordy altercations ensued between the parties and P4 insisted that he would take away the entire pullas.In the fight, the accused persons started inflicting injuries to the complainant, and his sons P5 (A-2) gave a pharsi blow on the head of P2, P4 (A-1) caused injury to P1 (PW-5) with two jaily blows. Additionally, P7 and P6 attacked the complainant with lathi blows on shoulder and left elbow respectively and caused several other injuries to the complainant party. P1 and his injured sons raised alarm, hearing which P9 and P10 came to rescue them and on seeing them, the accused persons fled away. The injured witnesses were taken to the Primary Health Centre where Dr. D1, Medical Officer, medically examined the injured persons. Injured P2 was vomiting in the hospital and later on he was referred to General Hospital, Gurgaon as his condition deteriorated. A CT scan disclosed that large extra-dural haematoma was found in the frontal region with mass effect and P2 needed urgent surgery and he was operated upon and the large extra-dural haematoma was removed. Dr. D1 (PW-2) also examined the other injured persons, PW 5-P1 and PW 6- P3.4. Statement of P1 was recorded, based on which F.I.R. was registered at Police Station. PW 8 P8 (ASI) had taken up the investigation. He examined the witnesses and after completion of investigation and challan was filed. In the trial court, prosecution examined nine witnesses including P1-PW5, P3-PW6 and Dr. D2-PW2 and Dr. D3-PW9, Neuro Surgeon, PW8-investigating officer and other witnesses. The accused were examined about the incriminating evidence and circumstances.First accused P4 pleaded that on the date of occurrence-complainant party P1 and his sons P3 and P2 forcibly trespassed into the land belonging to the accused and attempted to forcibly cut the pullas. P1 further claims that he along with P6 caused injuries to the complainant party in exercise of right of private defence of property. He has denied that P9 and P10 had seen the incident. P5 (A-2) and P7 (A-3) stated that they were not present on the spot and they have been falsely implicated. P6 (A-4) adopted the stand of his father P4.5. Upon consideration of oral and documentary evidence, the learned Additional Sessions Judge vide judgment dated 17.2.2000 convicted all the accused persons and sentenced them to undergo rigorous imprisonment for five years and one year respectively and a fine of Rs. 500/- each with default clause. Aggrieved by the said judgment, the accused-appellants filed criminal appeal before the High Court. The High Court vide impugned judgment dated 19.8.2011 modified the judgment of the trial court thereby convicted P4 (A-1) and sentenced him to undergo rigorous imprisonment for one year, convicted second accused P5 and imposed sentence of imprisonment for five years as well the fine of Rs.500/- was confirmed by the High Court. He was sentenced to undergo six months rigorous imprisonment. Both the sentences were ordered to run concurrently. High Court modified the sentence of P7 (A-3) P6 (A-4) and sentenced them to undergo rigorous imprisonment for six months (two counts) respectively. In this appeal, the appellants assail the correctness of the impugned judgment.",
    "Represent this sentence for searching relevant passages: Assailing the legal acceptability of the judgment and order passed by the High Court where it has given endorsement to the judgment passed by the learned Additional Sessions Judge wherein the learned trial Judge had found the appellants guilty of the offences and imposed the sentence of rigorous imprisonment of seven years and a fine of Rs.1,000/- on the first score, five years rigorous imprisonment and a fine of Rs.1,000/- on the second score, eighteen months rigorous imprisonment and a fine of Rs.500/- on the third count and six months rigorous imprisonment and a fine of Rs.250/- on the fourth count with the default clause for the fine amount in respect of each of the offences. The learned trial Judge stipulated that all the sentences shall be concurrent. Filtering the unnecessary details, the prosecution case, in brief, is that the marriage between the appellant No. 1 and deceased  P1, sister of the informant, PW-2, was solemnized on 24.9.1997. After the marriage the deceased stayed with her husband and the mother-in-law, the appellant No.2 herein, at the matrimonial home. In the wedlock, two children, one son and a daughter were born. On 11.9.2001, the informant, brother of the deceased, got a telephonic call from the accused No. 1 that his sister  P1 had committed suicide. On receipt of the telephone call came along with his friend, P2, PW-20, and at that juncture, the husband of  P1,  P3, informed that the deceased was fed up with the constant ill-health of her children and the said frustration had led her to commit suicide by tying a 'dupatta' around her neck. The brother of the deceased did not believe the version of  P3, and lodged an FIR alleging that the husband and the mother-in-law of the deceased, after the marriage, had been constantly asking for dowry of Rs.2 lacs from the father of the deceased, but as the said demand could not be satisfied due to the financial condition of the father, the husband and his mother started ill-treating her in the matrimonial home and being unable to tolerate the physical and mental torture she was compelled to commit suicide. Be it noted, as the death was unnatural, the police had sent the dead body for post mortem and the doctor conducting the autopsy opined that the death was due to suicide. After the criminal law was set in motion on the base of the FIR lodged by the brother, the investigating officer examined number of witnesses and after completing all the formalities laid the charge sheet before the competent Court, who, in turn, committed the matter to the Court of Session. The accused persons denied the allegations and claimed to be tried. The prosecution, in order to establish the charges levelled against the accused persons, examined 22 witnesses and got marked number of documents. The defence chose not to adduce any evidence. 4. The learned trial Judge principally posed four questions, namely, whether the accused persons had inflicted unbearable torture on the deceased as well as caused mental harassment to make themselves liable for punishment; whether the material brought on record established the offence; whether the physical and mental torture on the deceased compelled her to commit suicide on 11.9.2001 as a consequence of which the accused persons had become liable to be convicted; and whether the accused persons had demanded a sum of Rs.2 lacs towards dowry from the parents of  P1 so as to be found guilty. The learned trial Judge answered all the questions in the affirmative and opined that the prosecution had been able to prove the offences to the hilt and, accordingly, imposed the sentence as stated hereinbefore. Grieved by the judgment of conviction and the order of sentence the appellants filed an appeal. The High Court at the stage of admission had suo motu issued notice for enhancement of sentence. The State had appealed for the self-same purpose. The appeals and the revision application were disposed of by a common judgment dated 6.9.2007 whereby the Division Bench of the High Court concurred with the view expressed by the learned trial Judge and, accordingly, dismissed the appeals preferred by the accused as well as by the State and resultantly suo motu by the High Court also stood dismissed. The non-success in the appeal has compelled the accused-appellants to prefer this appeal by special leave."
]
OUT_JSON = "bge_embeddings.json"

# Conectar a Milvus (opcional)
MILVUS = True
if MILVUS:
    connections.connect("default", host="localhost", port="19530")

model = SentenceTransformer(MODEL_NAME)

def load_texts(folder, prefix):
    data = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            data.append({"id": os.path.splitext(fname)[0], "text": txt, "prefix": prefix})
    return data

casedocs = load_texts(CASEDOC_DIR, "")   # documentos sin prefijo
statutes = load_texts(STATUTE_DIR, "")   # idem

# Embeddings documentos
doc_texts = [d["text"] for d in casedocs]
stat_texts = [s["text"] for s in statutes]

doc_emb = model.encode(doc_texts, normalize_embeddings=NORMALIZE)
stat_emb = model.encode(stat_texts, normalize_embeddings=NORMALIZE)

# Embeddings queries (ya incluyen prefijo en la lista QUERIES)
query_emb = model.encode(QUERIES, normalize_embeddings=NORMALIZE)

dim = doc_emb.shape[1]

# Construir estructura JSON para queries
query_payload = {
  "model": MODEL_NAME,
  "dimension": dim,
  "normalized": NORMALIZE,
  "queries": [
    {"query_text": q, "vector": query_emb[i].tolist()}
    for i, q in enumerate(QUERIES)
  ]
}

query_json = "queries_embeddings.json"
with open(query_json, "w", encoding="utf-8") as f:
  json.dump(query_payload, f, ensure_ascii=False)
print(f"JSON de queries creado: {query_json} (dim={dim})")

# Construir estructura JSON para casedocs
casedocs_payload = {
  "model": MODEL_NAME,
  "dimension": dim,
  "normalized": NORMALIZE,
  "casedocs": [
    {"id": casedocs[i]["id"], "vector": doc_emb[i].tolist()}
    for i in range(len(casedocs))
  ]
}

casedocs_json = "casedocs_embeddings.json"
with open(casedocs_json, "w", encoding="utf-8") as f:
  json.dump(casedocs_payload, f, ensure_ascii=False)
print(f"JSON de casedocs creado: {casedocs_json} (dim={dim})")

# Construir estructura JSON para statutes
statutes_vectors_list = [
  {"vector": stat_emb[i].tolist(), "textfile_id": statutes[i]["id"]}
  for i in range(len(statutes))
]

# Save the data to a JSON file
statutes_json = "statutes_embeddings.json"
with open(statutes_json, 'w', encoding='utf-8') as json_file:
  json.dump(statutes_vectors_list, json_file, indent=2, ensure_ascii=False)
  
print(f"JSON de statutes creado: {statutes_json} (dim={dim})")

# Milvus (colecciones: casedoc_bge_base_en_v1_5, statute_bge_base_en_v1_5, query_bge_base_en_v1_5)
def ensure_collection(name, dim):
    if utility.has_collection(name):
        return Collection(name)
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=16),
    ]
    schema = CollectionSchema(fields, description=name)
    col = Collection(name, schema)
    return col

if MILVUS:
    # Insert casedocs
    c_col = ensure_collection("casedoc_bge_base_en_v1_5", dim)
    if c_col.num_entities == 0:
        c_col.insert([
            [c["id"] for c in casedocs],
            doc_emb.tolist(),
            ["casedoc"] * len(casedocs)
        ])
        c_col.flush()
        c_col.create_index("embedding", {
            "index_type": "HNSW",
            "metric_type": "COSINE" if NORMALIZE else "IP",
            "params": {"M": 16, "efConstruction": 200}
        })

    # Insert statutes
    s_col = ensure_collection("statute_bge_base_en_v1_5", dim)
    if s_col.num_entities == 0:
        s_col.insert([
            [s["id"] for s in statutes],
            stat_emb.tolist(),
            ["statute"] * len(statutes)
        ])
        s_col.flush()
        s_col.create_index("embedding", {
            "index_type": "HNSW",
            "metric_type": "COSINE" if NORMALIZE else "IP",
            "params": {"M": 16, "efConstruction": 200}
        })

    # Insert queries
    q_col = ensure_collection("query_bge_base_en_v1_5", dim)
    if q_col.num_entities == 0:
        q_col.insert([
            [f"Q{i}" for i in range(len(QUERIES))],
            query_emb.tolist(),
            ["query"] * len(QUERIES)
        ])
        q_col.flush()
        q_col.create_index("embedding", {
            "index_type": "FLAT",
            "metric_type": "COSINE" if NORMALIZE else "IP",
            "params": {}
        })

    # Ejemplo de búsqueda
    s_col.load()
    vec = query_emb[0]
    hits = c_col.search(
        data=[vec],
        anns_field="embedding",
        param={"metric_type": "COSINE" if NORMALIZE else "IP", "params": {"ef": 64}},
        limit=5,
        output_fields=["id","kind"]
    )
    print("Top 5 casedocs para primera query:")
    for h in hits[0]:
        print(h.id, h.distance)

    # Liberar
    c_col.release()
