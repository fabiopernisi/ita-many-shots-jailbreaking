import deepl
import pandas as pd
import logging
import fire
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_translator():
    """Retrieve the DeepL translator using the API key from the environment variables."""
    auth_key = os.getenv('DEEPL_AUTH_KEY')
    if not auth_key:
        raise ValueError("No DeepL API key found in environment variables")
    return deepl.Translator(auth_key)

def translate_text(translator, text, target_lang="IT"):
    """Translate a given text to the specified target language using the provided translator."""
    try:
        result = translator.translate_text(text, target_lang=target_lang)
        return result.text
    except deepl.APIError as e:
        logging.error(f"DeepL API error occurred: {str(e)}")
        return None

def translate_dataset(input_file, output_file, columns_to_translate, target_lang="IT"):
    """Translate specified columns of a dataset and save the updated DataFrame with new columns."""
    translator = get_translator()  # Get the translator
    try:
        df = pd.read_csv(input_file)
        translated_df = df.copy()

        # Ensure columns_to_translate is treated as a list of strings
        if isinstance(columns_to_translate, tuple):
            columns_list = [col.strip() for col in columns_to_translate]  # Convert tuple to list and strip spaces
        else:
            columns_list = [col.strip() for col in columns_to_translate.split(',')]  # Split string into list and strip spaces

        for column in columns_list:
            translated_column_name = f"{column}_{target_lang}"
            translated_df[translated_column_name] = df[column].apply(lambda text: translate_text(translator, text, target_lang))
        
        translated_df.to_csv(output_file, index=False)
        logging.info(f"Translations completed and saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")

def main(input_path, output_path, columns_to_translate, target_lang="IT"):
    translate_dataset(input_path, output_path, columns_to_translate, target_lang)

if __name__ == "__main__":
    fire.Fire(main)
