import math
from collections import Counter
from preprocessing import preprocess 


def cosine_similarity(query_vector, document_vector):
    """
    Calculates the cosine similarity between a query vector and a document vector.

    Args:
        query_vector (dict): The query vector as a dictionary with terms as keys and weights as values.
        document_vector (dict): The document vector as a dictionary with terms as keys and weights as values.

    Returns:
        float: The cosine similarity between the query vector and the document vector.
    """
    try:
        dot_product = sum(query_vector[i] * document_vector[i] for i in range(len(query_vector)))
        query_norm = math.sqrt(sum(query_vector[i] ** 2 for i in range(len(query_vector))))
        document_norm = math.sqrt(sum(document_vector[i] ** 2 for i in range(len(document_vector))))
        similarity = dot_product / (query_norm * document_norm)
        return similarity
    except ZeroDivisionError:
        print("Error: Division by zero occurred.")
    except KeyError as e:
        print(f"Error: Key '{e.args[0]}' not found in one of the input vectors.")
    except TypeError:
        print("Error: Invalid input vector format. Must be a dictionary with terms as keys and weights as values.")





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
                # idf = math.log10(total_documents / len(inverted_index[term]))  # Inverse document frequency
                idf=len(inverted_index[term])
                tfidf = tf * idf  # TF-IDF value
                tfidf_vector[j-1] = tfidf  # Update vector at index j-1
            # else:
            #     tfidf_vector[j-1]=0

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
    query_terms=preprocess(query)
    print(query_terms)

    # Calculate TF-IDF for each term in the query
    query_term_count = Counter(query_terms)
    print(query_term_count)
    for i, term in enumerate(inverted_index, start=1):
        if term in query_term_count:
            tf = query_term_count[term]  # Term frequency
            # idf = math.log10(total_documents / len(inverted_index[term]))  # Inverse document frequency
            idf=len(inverted_index[term])
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
    print(len(query_vector))
    print(len(documents_tfidf))
    documents = []

    # Calculate cosine similarity between query vector and document vectors
    for doc_id, doc_vector in documents_tfidf.items():
        cosine_sim = cosine_similarity(query_vector, doc_vector)
        print(cosine_sim)
        # Filter documents based on relevance threshold
        if cosine_sim >= alpha:
            tuple_doc=(doc_id,cosine_sim)
            documents.append(tuple_doc)
        else:
            pass

        documents.sort(key=lambda x: x[1],reverse=True)
        relevant_documents = [x[0] for x in documents]

    return relevant_documents
