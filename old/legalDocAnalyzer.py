# Universidad de La Laguna
# Escuela Superior de Ingenieria y Tecnologia
# Grado en Ingenieria Informatica
# Asignatura: Trabajo Fin de Grado
# Autor: Javier Garcia Santana
# Correo: alu0101391663@ull.edu.es
# Fecha: 10/03/2025
#
# Archivo legalDocAnalyzer.py: Este programa se utiliza para leer y cargar documentos legales desde archivos de texto. 
# Proporciona funciones para leer el contenido de un archivo y cargar múltiples archivos 
# en un diccionario, categorizados como "casedocs" o "statutes".

# Enlaces de interes: https://www.kaggle.com/datasets/ananyapam7/legalai/data
#
# Historial de revisiones
# 10/03/2025 - Creacion (primera version) del codigo

import os

def read_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as file:  
    content = file.read()  
  return content



def process_query_to_dict(line):
    parts = line.strip().split("||", 1)  # Split at '||', only once
    if len(parts) == 2:  # Ensure there are exactly 2 parts
        key, value = parts
        return {key: value}
    return {}  

def process_judgment_to_dict(line, result_dict):
    parts = line.strip().split()  # Split by whitespace
    if len(parts) != 4:  # Ensure there are exactly 4 columns
        return  # Ignore incorrectly formatted lines

    key, attr1, attr2, last_col = parts  # Unpack columns

    if last_col == "1":  # Check if last column is "1"
        if key in result_dict:
            result_dict[key].append((attr1, attr2))  # Append new values
        else:
            result_dict[key] = [(attr1, attr2)]  # Create a new list


def load_objects(type, folder_path):
  docs = {}  
  if type == "casedocs":
    range_objects = range(1, 2915)
    obj_name = "C"  
  elif type == "statutes":
    range_objects = range(1, 201)
    obj_name = "S" 

  for i in range_objects:  
      file_name = f"{obj_name}{i}.txt"
      file_path = os.path.join(folder_path, file_name)  
      
      if os.path.exists(file_path):  # Check if file exists
          case_name = os.path.splitext(file_name)[0]  # Get filename without extension
          docs[case_name] = read_file(file_path)  # Read file and store in dictionary
  return docs  


def load_queries(file_path): 
  queries = {}
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = file.readline() 
        queries.update(process_query_to_dict(line))
      
  return queries

def load_judgments(file_path):
  judgments = {}
  with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = file.readline() 
        process_judgment_to_dict(line, judgments)
      
  return judgments
            

  
cases = load_objects("casedocs", "./archive/Object_casedocs")  
statutes = load_objects("statutes", "./archive/Object_statutes")
queries = load_queries("./archive/Query_doc.txt")
r_priorcases = load_judgments("./archive/relevance_judgments_priorcases.txt")
r_statutes = load_judgments("./archive/relevance_judgments_statutes.txt")


print("Case 1: " + str(cases["C1"]))
print("Statute 200: " + str(statutes["S200"]))
print("Query Q22: " + str(queries["AILA_Q22"]) + "\n")
print("Relevance prriorcase Q4: " + str(r_priorcases["AILA_Q4"]) + "\n") 
print("Relevance statute Q47" + str(r_statutes["AILA_Q47"])+ "\n") 
