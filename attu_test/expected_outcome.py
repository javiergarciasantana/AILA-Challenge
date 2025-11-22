import re

def expected_outcome(input_type, query_id, doc_id):
  """
  Process the input file based on the input type (statute or priorcase) 
  and query ID to match specific lines.

  Args:
    input_type (str): Either "statute" or "priorcase".
    query_id (str): The query ID or case/statute ID to match in the regex.

  Returns:
    None
  """
  # Determine the input file based on the input type
  if input_type == "statute":
    input_file = "../archive/relevance_judgments_statutes.txt"
  elif input_type == "casedoc":
    input_file = "../archive/relevance_judgments_priorcases.txt"
  else:
    raise ValueError("Invalid input type. Use 'statute' or 'priorcase'.")

  # Open the file and process it
  with open(input_file, "r") as file:
    for line in file:
      # Match lines based on the query ID and input type
      pattern = rf"^{query_id} Q0 {doc_id} 1$"
      if re.match(pattern, line.strip()):
        print(line.strip())