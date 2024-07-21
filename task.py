import pickle
from nltk.stem import PorterStemmer
import os
from nltk.tokenize import word_tokenize
import math
from collections import Counter


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

    print("Text List:", text_list)  # Print the resulting list of text contents
    return text_list


# def calculate_tf(term, document):
#     """
#     Calculates the term frequency (TF) of a term in a document.

#     Args:
#         term (str): The term whose TF needs to be calculated.
#         document (list): The list of preprocessed tokens in the document.

#     Returns:
#         float: The term frequency of the term in the document.
#     """
#     tf = document.count(term) / len(document)
#     return tf


# def calculate_idf(term, inverted_index, total_documents):
#     """
#     Calculates the inverse document frequency (IDF) of a term in the collection.

#     Args:
#         term (str): The term whose IDF needs to be calculated.
#         inverted_index (dict): The inverted index of the collection.
#         total_documents (int): The total number of documents in the collection.

#     Returns:
#         float: The inverse document frequency of the term in the collection.
#     """
#     document_frequency = len(inverted_index.get(term, []))
#     idf = math.log10(total_documents / (document_frequency + 1))
#     return idf


# def calculate_tf_idf(term, document, inverted_index, total_documents):
#     """
#     Calculates the TF-IDF (Term Frequency-Inverse Document Frequency) of a term in a document.

#     Args:
#         term (str): The term whose TF-IDF needs to be calculated.
#         document (list): The list of preprocessed tokens in the document.
#         inverted_index (dict): The inverted index of the collection.
#         total_documents (int): The total number of documents in the collection.

#     Returns:
#         float: The TF-IDF of the term in the document.
#     """
#     tf = calculate_tf(term, document)
#     idf = calculate_idf(term, inverted_index, total_documents)
#     tf_idf = tf * idf
#     return tf_idf


def cosine_similarity(query_vector, document_vector):
    """
    Calculates the cosine similarity between a query vector and a document vector.

    Args:
        query_vector (dict): The query vector as a dictionary with terms as keys and weights as values.
        document_vector (dict): The document vector as a dictionary with terms as keys and weights as values.

    Returns:
        float: The cosine similarity between the query vector and the document vector.
    """
    dot_product = sum(query_vector[i] * document_vector[i] for i in range(len(query_vector)))
    query_norm = math.sqrt(sum(query_vector[i] ** 2 for i in range(len(query_vector))))
    document_norm = math.sqrt(sum(document_vector[i] ** 2 for i in range(len(document_vector))))
    similarity = dot_product / (query_norm * document_norm)
    return similarity






def calculate_tfidf_document(documents, inverted_index, total_documents):
    """
    Calculates the TF-IDF representation of documents.

    Args:
        documents (list): List of document texts.
        inverted_index (dict): The inverted index of terms.
        total_documents (int): Total number of documents.

    Returns:
        dict: The dictionary containing the TF-IDF representation of documents.
    """
    documents_tfidf = {}

    for i, doc in enumerate(documents, start=1):
        doc_id = str(i)  # Document ID as string
        doc=preprocess(doc)
        tfidf_vector = [0] * len(inverted_index)  # Initialize vector with zeros

        # Calculate TF-IDF for each term in the document
        for j, term in enumerate(inverted_index, start=1):
            if term in doc:
                tf = doc.count(term)  # Term frequency
                idf = math.log10(total_documents / len(inverted_index[term]))  # Inverse document frequency
                tfidf = tf * idf  # TF-IDF value
                tfidf_vector[j-1] = tfidf  # Update vector at index j-1

        documents_tfidf[doc_id] = tfidf_vector

    return documents_tfidf


def get_query_vector(query, inverted_index, total_documents):
    """
    Calculates the TF-IDF vector representation for a query.

    Args:
        query (str): The input query.
        inverted_index (dict): The inverted index of terms.
        total_documents (int): Total number of documents.

    Returns:
        list: The TF-IDF vector representation for the query.
    """
    query_vector = [0] * len(inverted_index)  # Initialize vector with zeros

    # Tokenize query and perform case-folding
    # query_terms = query.lower().split()
    query_terms=preprocess(query)
    print(query_terms)
    # Remove stopwords from query terms
    # query_terms = [term for term in query_terms if term not in STOPWORDS]

    # Perform stemming on query terms
    # query_terms = [STEMMER.stem(term) for term in query_terms]

    # Calculate TF-IDF for each term in the query
    query_term_count = Counter(query_terms)
    print(query_term_count)
    for i, term in enumerate(inverted_index, start=1):
        if term in query_term_count:
            tf = query_term_count[term]  # Term frequency
            idf = math.log10(total_documents / len(inverted_index[term]))  # Inverse document frequency
            tfidf = tf * idf  # TF-IDF value
            query_vector[i-1] = tfidf  # Update vector at index i-1

    return query_vector


def query_processing(query, inverted_index, documents_tfidf, alpha=0.05):
    """
    Processes a query and retrieves relevant documents from the vector space model.

    Args:
        query (str): The input query.
        inverted_index (dict): The inverted index of terms.
        documents_tfidf (dict): The dictionary containing the TF-IDF representation of documents.
        alpha (float): The threshold for relevance filtering.

    Returns:
        list: The list of relevant documents.
    """
    query_vector = get_query_vector(query, inverted_index, len(documents_tfidf))
    input()
    print(len(query_vector))
    print(len(documents_tfidf))
    relevant_documents = []

    # Calculate cosine similarity between query vector and document vectors
    for doc_id, doc_vector in documents_tfidf.items():
        cosine_sim = cosine_similarity(query_vector, doc_vector)
        print(cosine_sim)
        # Filter documents based on relevance threshold
        if cosine_sim >= alpha:
            relevant_documents.append(doc_id)

    return relevant_documents






# Example usage:
documents_path = r"C:\Users\LENOVO\Desktop\IR_assignment_2\Dataset"  # Path to the folder containing the text documents
inverted_index = create_inverted_index(documents_path)
# print(inverted_index)
documents=get_documents(documents_path)
print(len(documents))
documents_tfidf=calculate_tfidf_document(documents,inverted_index,len(documents))

print(query_processing('cricket politics',inverted_index,documents_tfidf,0.05))
