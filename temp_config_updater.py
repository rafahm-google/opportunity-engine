
import os
import json

def anonymize_config_files():
    """
    Renames config files to a generic name and updates their content
    to remove brand-specific information.
    """
    
    # Mapping from old brand directory to new generic directory and name
    anonymization_map = {
        "advertiser_b": {"old_name": "claro", "generic_name": "Advertiser B"},
        "advertiser_c": {"old_name": "estacio", "generic_name": "Advertiser C"},
        "advertiser_d": {"old_name": "itau", "generic_name": "Advertiser D"},
        "advertiser_f": {"old_name": "vivo", "generic_name": "Advertiser F"}
    }

    base_path = "inputs"
    print("="*50 + "\nAnonymizing config files and their content...\n" + "="*50)

    for generic_dir, info in anonymization_map.items():
        try:
            current_dir_path = os.path.join(base_path, generic_dir)
            old_config_name = f"config_{info['old_name']}.json"
            new_config_name = "config.json"
            old_config_path = os.path.join(current_dir_path, old_config_name)
            new_config_path = os.path.join(current_dir_path, new_config_name)

            # Rename the config file if it exists
            if os.path.exists(old_config_path):
                os.rename(old_config_path, new_config_path)
                print(f"   - Renamed: {old_config_path} -> {new_config_path}")

                # Update the content of the newly renamed file
                with open(new_config_path, 'r') as f:
                    config_data = json.load(f)

                config_data["advertiser_name"] = info["generic_name"]
                
                # Update file paths within the config
                for key, value in config_data.items():
                    if isinstance(value, str) and f"inputs/{info['old_name']}/" in value:
                        config_data[key] = value.replace(f"inputs/{info['old_name']}/", f"inputs/{generic_dir}/")

                with open(new_config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                print(f"   - Updated content for: {new_config_path}")

            # Handle special cases for 'estacio'
            if info['old_name'] == 'estacio':
                for suffix in ['_temp', '_youtube']:
                    old_special_name = f"config_estacio{suffix}.json"
                    new_special_name = f"config{suffix}.json"
                    old_special_path = os.path.join(current_dir_path, old_special_name)
                    new_special_path = os.path.join(current_dir_path, new_special_name)
                    if os.path.exists(old_special_path):
                        os.rename(old_special_path, new_special_path)
                        print(f"   - Renamed: {old_special_path} -> {new_special_path}")
                        # Also update content for these files
                        with open(new_special_path, 'r') as f:
                            config_data = json.load(f)
                        config_data["advertiser_name"] = info["generic_name"]
                        for key, value in config_data.items():
                            if isinstance(value, str) and f"inputs/estacio/" in value:
                                config_data[key] = value.replace(f"inputs/estacio/", f"inputs/{generic_dir}/")
                        with open(new_special_path, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        print(f"   - Updated content for: {new_special_path}")

        except Exception as e:
            print(f"   - ❌ ERROR processing directory {generic_dir}. Details: {e}")
            
    print("\n" + "="*50 + "\n✅ Anonymization of config files complete.\n" + "="*50)

if __name__ == "__main__":
    anonymize_config_files()
