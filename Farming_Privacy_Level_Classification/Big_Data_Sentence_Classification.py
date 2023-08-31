"""
Extract sentences from document and classify

Varun Mandalapu
"""
import csv
import chardet
import spacy
from pdfminer.high_level import extract_text
import re


def detect_encoding(file_path):
    """
    Detects the encoding of a given file.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        str: The detected encoding of the file.
    """
    with open(file_path, 'rb') as f:
        chunk = f.read(1024)

    result = chardet.detect(chunk)
    return result['encoding']

def remove_nonprintable(text):
    """
    Removes non-printable characters from the given text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The cleaned text with non-printable characters removed.
    """
    cleaned_text = re.sub(r'\(cid:\d+\)', '', text)
    return cleaned_text

def preprocess_text(text):
    """
    Preprocesses the given text by replacing line breaks with spaces.

    Parameters:
        text (str): The input text.

    Returns:
        str: The preprocessed text with line breaks replaced by spaces.
    """
    return text.replace('\n', ' ')

def preprocess_sentences(sentences):
    """
    Preprocesses a list of sentences by removing extra spaces between sentences.

    Parameters:
        sentences (list of str): A list of sentences.

    Returns:
        list of str: The preprocessed sentences with extra spaces removed.
    """
    return [re.sub(r'\.\s+', '. ', sentence.strip()) for sentence in sentences]

def split_sentences(text):
    """
    Splits the given text into sentences based on periods followed by spaces, excluding specific abbreviations.

    Parameters:
        text (str): The input text.

    Returns:
        list of str: The sentences extracted from the text.
    """
    # Remove text between parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    # Manually split text into sentences based on periods followed by spaces, excluding specific abbreviations
    return re.split(r'\.(?!\s*(?:etc|i\.e|e\.g)\.\s*)', text)

def classify_sentences_with_words(pdf_path, class_words):
    """
    Classifies sentences in a given PDF file based on the presence of specific words for each class.

    Parameters:
        pdf_path (str): The path to the PDF file.
        class_words (dict): A dictionary mapping class names to lists of class-defining words.

    Returns:
        list of tuple: A list of tuples containing the class name, the classified sentence, and observed words.
    """
    # Detect the encoding of the PDF
    encoding = detect_encoding(pdf_path)

    # Load the spaCy language model with the detected encoding
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"], exclude=["tok2vec", "morphologizer"])
    nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, rules={}, prefix_search=None, suffix_search=None)

    # Read text from the PDF using the detected encoding
    pdf_text = extract_text(pdf_path, codec=encoding)

    # Remove non-printable characters
    pdf_text_cleaned = remove_nonprintable(pdf_text)

    # Preprocess the text to replace line breaks with spaces
    pdf_text_preprocessed = preprocess_text(pdf_text_cleaned)

    # Split the text into sentences manually
    sentences = split_sentences(pdf_text_preprocessed)

    # Initialize a list to store individual sentences classified by class
    classified_sentences = []

    # Iterate through each sentence
    for sentence in sentences:
        # Check if any of the class words are present in the sentence
        for class_name, words in class_words.items():
            observed_words = [word.lower() for word in words if f" {word.lower()} " in f" {sentence.lower()} "]
            if observed_words:
                classified_sentences.append((class_name, sentence.strip(), ', '.join(observed_words)))

    return classified_sentences

if __name__ == "__main__":
    # Replace "your_pdf_file.pdf" with the path to your PDF document
    pdf_file = "eu_main_document/processed_eu_document.pdf"

    # Define the class words and their corresponding class names
    class_words = {
        "Obl": ["shall be required", "will be required", "shall be obligated", "shall", "must", "will", "have to", "should", "ought to have", "will be paid", "shall be paid",
               "agree", "agrees", "anknowledges", "acknowledge", "represents and warrants", "shall be responsible for", "will be responsible for"],
        "Ent": ["shall be entitled", "will be entitled", "shall be paid", "will be paid", "shall retain", "will retain", 
                "will receive", "shall receive", "shall have the right to", "shall be retained", "shall be kept",
               "shall be claimed", "shall be accessible", "shall be owned", "shall be determined", "agrees", "shall be entitles to", "represents and warrants", "acknowledges",
               "waives no rights", "retains all other rights", "will be entitles to"],
        "Pro": ["shall not", "will not", "must not", "may not", "cannot", "shall have no right", "can not", 
                "shall not be allowed", "will not be allowed", "shall not assist", "shall be prohibited", "will be prohibited", "nor shall", "not to be",
               "neither lessor nore lessee may", "in no event shall", "nor will", "will not allow", "nor may"],
        "Per": ["shall be permitted", "shall also be permitted", "can", "may", "could", "shall be allowed", "will be allowed", "is permitted", "will allow",
               "has the right", "or at landlord's option", "shall be permitted to"],
        "Nobl": ["shall not be liable for", "will not be liable for", "shall not be obligated to", "will not be obligated to", "shall not be obligated for", "will not be obligated for",
                 "shall not be responsible for", "will not be responsible for", "shall not be required to", "will not be required to", "shall not be liable",
                "shall have no obligation to", "in no event shall landlord be obligated to", "waives", "shall have no liability"],
        "Nent": ["shall not be entitled to", "will not be entitled to", "shall not have the right to", "will not have the right to", "shall not be entitled for",
                "will not be entitled for", "shall have no right to", "waives no rights", "shall have no obligation to", "waives", "shall not be required", "shall not be obligated",
                "waive the right", "shall not have the right to"]
        # Add more classes and their corresponding words as needed
    }

    classified_sentences = classify_sentences_with_words(pdf_file, class_words)
        
    # Write the data to a CSV file
    with open('classified_output/classified_sentences.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Class', 'Sentence', 'Observed_Word']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for class_name, sentence, observed_words in classified_sentences:
            writer.writerow([class_name, sentence, observed_words])

    print("CSV file created successfully.")

