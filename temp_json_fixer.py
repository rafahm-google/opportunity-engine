
import os
import json

def fix_json_trailing_commas(directories):
    """
    Finds all config.json files in the given directories and removes trailing commas.
    """
    print("="*50 + "\nFixing trailing commas in JSON files...\n" + "="*50)
    for directory in directories:
        config_path = os.path.join(directory, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    # Reading the raw text
                    raw_content = f.read()
                
                # A simple but effective way to remove trailing commas before a '}'
                # This is safer than complex regex for this specific problem
                cleaned_content = raw_content.replace(',\n}', '\n}')
                
                # Validate that it's now valid JSON
                json.loads(cleaned_content)
                
                # Write the cleaned content back
                with open(config_path, 'w') as f:
                    f.write(cleaned_content)
                
                print(f"   - ✅ Fixed and validated: {config_path}")

            except Exception as e:
                print(f"   - ❌ ERROR fixing {config_path}. Details: {e}")
    print("\n" + "="*50 + "\n✅ JSON fixing complete.\n" + "="*50)

if __name__ == "__main__":
    dirs_to_check = [
        "inputs/advertiser_b",
        "inputs/advertiser_c",
        "inputs/advertiser_d",
        "inputs/advertiser_f"
    ]
    fix_json_trailing_commas(dirs_to_check)
