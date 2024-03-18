import numpy as np
import numpy.typing as npt
from collections import Counter
from typing import Tuple, List, Set
import pandas as pd


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    # Flatten the list of sentences into a single list of words
    words = [word for sentence in sentences for word in sentence.split()]

    # Count the frequency of each word
    word_counts = Counter(words)

    # Replace words with frequency less than 2 with "<UNK>"
    sentences = [' '.join('<UNK>' if word_counts[word] < 2 else word for word in sentence.split()) for sentence in
                     sentences]


    # Tokenize and create a vocabulary using a set
    vocabulary = set()
    for sentence in sentences:
        words = sentence.split()
        vocabulary.update(words)


    # Convert the set to a sorted list for consistent order
    unique_words = sorted(list(vocabulary))

    # Create a dictionary to store word counts for each sentence
    word_counts = {word: [sentence.split().count(word) for sentence in sentences] for word in unique_words}


    # Convert the dictionary to a DataFrame
    bag_of_words_df = pd.DataFrame(word_counts)
    bag_of_words_df = bag_of_words_df.T


    # Display the updated DataFrame
    # print("Updated Bag-of-Words Matrix:")
    # print(bag_of_words_df.head(10))

    bag_of_words_matrix = bag_of_words_df.values



    return bag_of_words_matrix
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE

    intents, unique_intents = data

    intent_matrix = np.zeros((len(unique_intents), len(intents)), dtype=int)

    # Iterate over sentences and unique intents
    for sentence_idx, sentence_intent in enumerate(intents):
        for class_idx, unique_intent in enumerate(unique_intents):
            # Check if the sentence belongs to the current intent class
            if sentence_intent == unique_intent:
                intent_matrix[class_idx, sentence_idx] = 1


    return intent_matrix
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    exp_z = np.exp(z)  # Subtracting max(z) for numerical stability
    return exp_z / np.sum(exp_z, axis=0)
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.maximum(0, z)
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return np.where(z > 0, 1, 0)
    #########################################################################