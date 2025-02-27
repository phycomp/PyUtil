from json import loads as jsnLoads, JSONDecodeError

def parseFhirResource(data, parent_key=""):
    """
    Recursively parse a FHIR resource and retrieve all keys.
    This function is resource-type agnostic and works for any FHIR resource.
    
    Parameters:
    - data: The FHIR resource in dictionary form (parsed JSON).
    - parent_key: The key of the parent element (used to track full key paths).
    
    Returns:
    - A list of full key paths in the FHIR resource.
    """
    keys = []
    
    # If the data is a dictionary, iterate over its key-value pairs
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            keys.append(full_key)
            # Recursively retrieve keys for nested structures
            keys.extend(parseFhirResource(value, full_key))
    
    # If the data is a list, iterate over the items and use indices as part of the key
    elif isinstance(data, list):
        for i, item in enumerate(data):
            full_key = f"{parent_key}[{i}]"
            keys.append(full_key)
            # Recursively retrieve keys for list elements
            keys.extend(parseFhirResource(item, full_key))
    
    return keys

# Example of handling different FHIR resources
def rtrvFhirKeys(fhir_json):
    """
    Parse and retrieve all keys from a FHIR resource, regardless of type.
    
    Parameters:
    - fhir_json: A FHIR resource in JSON format.
    
    Returns:
    - A list of all keys in the FHIR resource.
    """
    try:
        # Parse the FHIR resource JSON into a dictionary
        fhir_data = jsnLoads(fhir_json)
        
        # Use the recursive function to parse the FHIR resource and retrieve keys
        parsed_keys = parseFhirResource(fhir_data)
        
        return parsed_keys
    except JSONDecodeError:
        print("Error: The provided data is not a valid JSON.")
        return []
