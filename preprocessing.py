import pickle
from nltk.stem import PorterStemmer
import os
from nltk.tokenize import word_tokenize

def preprocess(text,stop_words_path='Stopword-List.txt'):
    """
    Preprocesses a text by performing casefolding, tokenization,
    stemming, and stopword removal.

    Args:
        text (str): The input text to preprocess.

    Returns:
        list: The preprocessed tokens.
    """
    # read the stopwords from the text file
    stop_word_list=[]
    f=open(stop_words_path,"r")
    lines=f.readlines()

    for line in lines:
        stop_word_list.append(line.strip())

    # Convert to lowercase (casefolding)
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Initialize Snowball stemmer for English
    stemmer = PorterStemmer()

    # Perform stemming
    tokens = [stemmer.stem(token) for token in tokens]

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_word_list]

    return tokens



def create_inverted_index(documents_path):
    """
    Creates an inverted index from a set of text documents.

    Args:
        documents_path (str): The path to the folder containing the text documents.

    Returns:
        dict: The inverted index where keys are terms and values are lists of document IDs
        where the term occurs.
    """
    inverted_index = {}

    # Loop through each document in the folder
    for filename in os.listdir(documents_path):
        with open(os.path.join(documents_path, filename), "r") as file:
            doc_id = int(filename.replace('.txt','') ) # Extract document ID from filename
            text = file.read()
            tokens = preprocess(text)

            # Update the inverted index with the document ID for each term
            for token in tokens:
                if token not in inverted_index:
                    inverted_index[token] = []
                inverted_index[token].append(doc_id)

    # Save indexes to file
    file_path_1=documents_path+'\\inverted_index.pkl'
    # file_path_2=folder_path+'\\positional_index.pkl'
    with open(file_path_1, 'wb') as f:
        pickle.dump(inverted_index, f)
    
    return inverted_index




def get_documents(documents_path):
    file_list = os.listdir(documents_path)  # Get a list of files in the folder
    text_list = []  # List to store the contents of the text files

    # Loop through the files in the folder
    for file_name in file_list:
        if file_name.endswith(".txt"):  # Filter for text files only
            file_path = os.path.join(documents_path, file_name)  # Create the file path
            with open(file_path, "r") as file:  # Open the file in read mode
                text = file.read()  # Read the contents of the file
                position = int(file_name.split(".")[0]) - 1  # Extract the position from the file name (assuming files are numbered starting from 1)
                text_list.insert(position, text)  # Insert the contents at the specified position in the list

    return text_list
