import os
import json
import shutil

def anonymize_directory_structure():
    """
    Anonymizes the directory and file structure within the temp folder.
    """
    base_path = "inputs_temp"
    
    anonymization_map = {
        "chevrolet": "advertiser_a",
        "claro": "advertiser_b",
        "estacio": "advertiser_c",
        "itau": "advertiser_d",
        "neosaldina": "advertiser_e",
        "vivo": "advertiser_f"
    }

    print("="*50 + "\nAnonymizing directory and file names in 'inputs_temp/'...\n" + "="*50)
    
    for old_name, new_name in anonymization_map.items():
        old_path = os.path.join(base_path, old_name)
        new_path = os.path.join(base_path, new_name)
        if os.path.exists(old_path):
            # Use shutil.move which can handle non-empty directories
            shutil.move(old_path, new_path)
            print(f"   - Renamed directory: {old_path} -> {new_path}")

            # Now, rename config files inside the new directory
            for filename in os.listdir(new_path):
                if 'config' in filename and old_name in filename:
                    new_filename = filename.replace(old_name, "generic")
                    os.rename(os.path.join(new_path, filename), os.path.join(new_path, new_filename))
                    print(f"     - Renamed config file: {filename} -> {new_filename}")


def update_config_content():
    """
    Updates the content of the anonymized config files.
    """
    base_path = "inputs_temp"
    
    anonymization_map = {
        "advertiser_a": {"old_name": "chevrolet", "generic_name": "Advertiser A"},
        "advertiser_b": {"old_name": "claro", "generic_name": "Advertiser B"},
        "advertiser_c": {"old_name": "estacio", "generic_name": "Advertiser C"},
        "advertiser_d": {"old_name": "itau", "generic_name": "Advertiser D"},
        "advertiser_e": {"old_name": "neosaldina", "generic_name": "Advertiser E"},
        "advertiser_f": {"old_name": "vivo", "generic_name": "Advertiser F"}
    }

    print("\n" + "="*50 + "\nUpdating content of anonymized config files...\n" + "="*50)

    for generic_dir, info in anonymization_map.items():
        dir_path = os.path.join(base_path, generic_dir)
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if 'config' in filename and filename.endswith('.json'):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        with open(file_path, 'r') as f:
                            raw_content = f.read()
                        
                        cleaned_content = raw_content.replace(',\n}', '\n}').replace(',\n    }', '\n    }')
                        config_data = json.loads(cleaned_content)

                        config_data["advertiser_name"] = info["generic_name"]
                        for key, value in config_data.items():
                            if isinstance(value, str) and f"inputs/{info['old_name']}/" in value:
                                config_data[key] = value.replace(f"inputs/{info['old_name']}/", f"inputs/{generic_dir}/")

                        with open(file_path, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        print(f"   - Updated content for: {file_path}")

                    except Exception as e:
                        print(f"   - ❌ ERROR updating {file_path}. Details: {e}")


if __name__ == "__main__":
    anonymize_directory_structure()
    update_config_content()
    print("\n" + "="*50 + "\n✅ Anonymization of 'inputs_temp/' complete.\n" + "="*50)