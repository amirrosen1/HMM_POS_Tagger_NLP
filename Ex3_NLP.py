import nltk
import numpy as np
from nltk.corpus import brown
from collections import Counter
from sklearn.metrics import confusion_matrix


PSEUDO_WORD_THRESHOLD = 2  # Minimum frequency for a word to avoid being classified as pseudo-word


def process_and_divide_dataset(tagged_sents=None, split_ratio=0.9):
    """
    Processes the tagged sentences to clean the tags and divides them into training and evaluation datasets.

    Args:
        tagged_sents (list): A list of tagged sentences from a corpus. If None, downloads and uses the Brown corpus.
        split_ratio (float): The ratio of data to be used for training (default is 0.9).

    Returns:
        tuple: A tuple containing the training dataset and the evaluation dataset.
    """

    # Download the Brown corpus if tagged_sents is not provided
    if tagged_sents is None:
        nltk.download("brown")
        from nltk.corpus import brown
        tagged_sents = brown.tagged_sents(categories="news")

    processed_sentences = []
    for sentence in tagged_sents:
        updated_sentence = []
        for token, label in sentence:
            # Truncate the label at the first occurrence of '+' or '-'
            if '+' in label or '-' in label:
                label = label.split('+', 1)[0].split('-', 1)[0]
            updated_sentence.append((token, label))
        processed_sentences.append(updated_sentence)

    # Divide the processed data into training and evaluation datasets
    training_portion = int(split_ratio * len(processed_sentences))
    training_data = processed_sentences[:training_portion]
    evaluation_data = processed_sentences[training_portion:]

    return training_data, evaluation_data


def calculate_tag_statistics(training_data):
    """
    Calculates the frequency and relative proportion of each POS tag in the training dataset.

    Args:
        training_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.

    Returns:
        dict: A dictionary where each key is a tag, and the value is a tuple (count, proportion).
              - `count`: The number of times the tag appears in the dataset.
              - `proportion`: The fraction of the total tags that this tag represents.
    """

    # Flatten the list of all labels across all sentences
    # This prepares the data for counting tag occurrences.
    labels = [token_label[1] for sent in training_data for token_label in sent]
    label_counter = Counter(labels)
    total_label_count = sum(label_counter.values())
    label_stats = {label: (count, count / total_label_count) for label, count in label_counter.items()}
    return label_stats


def build_word_label_map(training_data):
    """
    Builds a mapping of word-label pairs to their occurrence count in the training dataset.

    Args:
        training_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.

    Returns:
        dict: A dictionary where each key is a (token, label) tuple, and the value is the count of occurrences.
    """

    word_label_map = {}
    for sent in training_data:
        for token, label in sent:
            if (token, label) in word_label_map:
                word_label_map[(token, label)] += 1
            else:
                word_label_map[(token, label)] = 1
    return word_label_map


def compute_mle(training_data):
    """
    Computes the Maximum Likelihood Estimate (MLE) for each word in the dataset, associating it with
    the most likely POS tag based on observed frequencies.

    Args:
        training_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.

    Returns:
        tuple: A tuple containing:
            - mle_map (dict): A dictionary where each key is a token, and the value is a tuple
              (most_likely_tag, log_probability).
                - `most_likely_tag`: The tag with the highest likelihood for the token.
                - `log_probability`: The log probability of the tag given the token.
            - tag_statistics (dict): The tag statistics returned by `calculate_tag_statistics`.
    """

    tag_statistics = calculate_tag_statistics(training_data)
    token_label_map = build_word_label_map(training_data)
    mle_map = dict()
    for pair, freq in token_label_map.items():
        token, label = pair
        # For each word-label pair, calculate the log-probability of the label given the word
        log_prob_label_given_token = np.log((tag_statistics[label][1] * freq) / tag_statistics[label][0])
        if token in mle_map:
            if log_prob_label_given_token > mle_map[token][1]:
                mle_map[token] = label, log_prob_label_given_token
        else:
            mle_map[token] = label, log_prob_label_given_token

    return mle_map, tag_statistics


def test_mle_error_rates(mle_map, evaluation_data):
    """
    Evaluates the performance of the Maximum Likelihood Estimation (MLE) model
    on the evaluation dataset and computes error rates.

    Args:
        mle_map (dict): A dictionary mapping tokens to their most likely POS tags and log probabilities.
                        Format: {token: (most_likely_tag, log_probability)}.
        evaluation_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.

    Returns:
        tuple: A tuple containing:
            - overall_error_rate (float): The error rate for all tokens.
            - known_error_rate (float): The error rate for tokens seen during training.
            - unknown_error_rate (float): The error rate for tokens not seen during training.
    """

    correct_predictions, total_predictions = 0, 0
    correct_known, total_known = 0, 0
    correct_unknown, total_unknown = 0, 0

    for sent in evaluation_data:
        for token, label in sent:
            if token in mle_map:
                predicted_label = mle_map[token][0]
                if predicted_label == label:
                    correct_predictions += 1
                    correct_known += 1
                total_known += 1
            else:
                predicted_label = 'NN'
                if predicted_label == label:
                    correct_predictions += 1
                    correct_unknown += 1
                total_unknown += 1
            total_predictions += 1

    overall_accuracy = correct_predictions / total_predictions
    known_accuracy = correct_known / total_known
    unknown_accuracy = correct_unknown / total_unknown
    return 1 - overall_accuracy, 1 - known_accuracy, 1 - unknown_accuracy


def build_emission_map(training_data, evaluation_data, apply_smoothing=False):
    """
    Builds an emission probability map for each (token, POS tag) pair in the dataset.

    Args:
        training_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.
        evaluation_data (list): A list of sentences from the evaluation dataset.
        apply_smoothing (bool): Whether to apply Add-One smoothing to account for unseen (token, label) pairs.

    Returns:
        dict: A dictionary mapping (token, label) pairs to their emission probabilities.
              Format: {(token, label): probability}.
    """

    tag_statistics = calculate_tag_statistics(training_data)
    token_label_map = build_word_label_map(training_data)
    unique_labels = set([token_label[1] for sent in training_data for token_label in sent])
    unique_tokens = set([token_label[0] for sent in training_data for token_label in sent])
    vocabulary_size = len(unique_tokens)

    if apply_smoothing:
        unique_tokens.update([token_label[0] for sent in evaluation_data for token_label in sent])

    for token in unique_tokens:
        for label in unique_labels:
            if apply_smoothing:
                if (token, label) not in token_label_map:
                    token_label_map[(token, label)] = 1
                else:
                    token_label_map[(token, label)] += 1
            else:
                if (token, label) not in token_label_map:
                    token_label_map[(token, label)] = 0

    emission_map = token_label_map
    for pair, frequency in emission_map.items():
        token, label = pair
        if apply_smoothing:
            emission_map[pair] = emission_map[pair] / (tag_statistics[label][0] + vocabulary_size)
        else:
            emission_map[pair] /= tag_statistics[label][0]

    return emission_map


def build_transition_map(training_data):
    """
    Builds a transition probability map for label bigrams (e.g., "NN -> VB") in the dataset.

    Args:
        training_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.

    Returns:
        dict: A dictionary mapping label bigrams to their transition probabilities.
              Format: {(label1, label2): probability}.
    """

    bigram_counter = Counter()
    unigram_counter = Counter()

    for sent in training_data:
        for idx in range(len(sent) - 1):
            token, label = sent[idx]
            unigram_counter[label] += 1
            bigram_counter[(label, sent[idx + 1][1])] += 1

            if idx == 0:
                bigram_counter[('START', label)] += 1
                unigram_counter['START'] += 1
            if idx == len(sent) - 2:
                bigram_counter[(sent[idx + 1][1], 'STOP')] += 1
                unigram_counter[sent[idx + 1][1]] += 1

    transition_map = {bigram: freq / unigram_counter[bigram[0]] for bigram, freq in bigram_counter.items()}
    return transition_map


def train_hidden_markov_model(training_data, evaluation_data, apply_smoothing=False):
    """
    Trains a Hidden Markov Model (HMM) by building emission and transition maps.

    Args:
        training_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.
        evaluation_data (list): A list of sentences from the evaluation dataset.
        apply_smoothing (bool): Whether to apply Add-One smoothing during the training process.

    Returns:
        tuple: A tuple containing:
            - emission_map (dict): The emission probabilities for (token, label) pairs.
            - transition_map (dict): The transition probabilities for (label1, label2) pairs.
    """

    emission_map = build_emission_map(training_data, evaluation_data, apply_smoothing)
    transition_map = build_transition_map(training_data)
    return emission_map, transition_map


def compute_viterbi_matrices(label_set, transition_map, emission_map, input_sentence):
    """
    Computes the transition and emission matrices for the Viterbi algorithm.

    Args:
        label_set (list): A list of all possible POS tags (labels).
        transition_map (dict): A dictionary mapping (label1, label2) pairs to their transition probabilities.
        emission_map (dict): A dictionary mapping (token, label) pairs to their emission probabilities.
        input_sentence (list): A list of tokens (words) in the input sentence.

    Returns:
        tuple: A tuple containing:
            - transition_matrix (np.ndarray): Transition probabilities between labels.
            - emission_matrix (np.ndarray): Emission probabilities for each (token, label) pair in the sentence.
            - start_vector (np.ndarray): Start probabilities for each label.
            - stop_vector (np.ndarray): Stop probabilities for each label.
    """

    num_words = len(input_sentence)
    num_labels = len(label_set)
    transition_matrix, emission_matrix = np.zeros((num_labels, num_labels)), np.zeros((num_labels, num_words))
    start_vector, stop_vector = np.zeros((num_labels, 1)), np.zeros((num_labels, 1))

    for i in range(num_labels):
        for j in range(num_labels):
            transition_matrix[i, j] = transition_map.get((label_set[i], label_set[j]), 0)
        start_vector[i] = transition_map.get(('START', label_set[i]), 0)
        stop_vector[i] = transition_map.get((label_set[i], 'STOP'), 0)

    for i in range(num_words):
        current_word = input_sentence[i][0]
        for j in range(num_labels):
            current_label = label_set[j]
            emission_matrix[j, i] = emission_map.get((current_word, current_label), 0.0)
            if not apply_smoothing and current_word not in vocabulary and current_label == 'NN':
                emission_matrix[j, i] = 1.0

    return transition_matrix, emission_matrix, start_vector, stop_vector


def execute_viterbi_algorithm(transition_map, emission_map, label_set, input_sentence):
    """
    Executes the Viterbi algorithm to determine the most probable sequence of POS tags for an input sentence.

    Args:
        transition_map (dict): A dictionary mapping (label1, label2) pairs to their transition probabilities.
        emission_map (dict): A dictionary mapping (token, label) pairs to their emission probabilities.
        label_set (list): A list of all possible POS tags (labels).
        input_sentence (list): A list of (token, label) tuples representing the input sentence.

    Returns:
        list: A list of predicted labels (POS tags) for the input sentence.
    """

    label_set = list(sorted(label_set))
    transition_matrix, emission_matrix, start_vector, stop_vector = compute_viterbi_matrices(
        label_set, transition_map, emission_map, input_sentence)

    num_words, num_labels = len(input_sentence), len(label_set)
    probability_matrix, backpointer_matrix = np.zeros((num_labels * num_words)).reshape(num_labels,
                                                                                        num_words), np.zeros(
        (num_labels * num_words)).reshape(num_labels, num_words)

    for step in range(num_words):
        previous_probabilities = probability_matrix[:, step - 1].reshape(num_labels, 1)
        emission_column = emission_matrix[:, step].reshape(num_labels, 1)
        for j in range(num_labels):
            if step == 0:
                current_probability = start_vector[j][0] * emission_column[j][0]
            else:
                transition_column = transition_matrix[:, j].reshape(num_labels, 1)
                current_probability = (previous_probabilities * transition_column) * emission_column[j][0]
            probability_matrix[j, step] = np.max(current_probability)
            backpointer_matrix[j, step] = np.argmax(current_probability)

    predicted_indices = []
    last_label_index = int(np.argmax(probability_matrix[:, -1].reshape(num_labels, 1) * stop_vector))
    predicted_indices.append(last_label_index)
    for step in range(num_words - 2, -1, -1):
        last_label_index = int(backpointer_matrix[last_label_index, step + 1])
        predicted_indices.append(last_label_index)

    predicted_labels = [label_set[idx] for idx in predicted_indices[::-1]]
    return predicted_labels


def evaluate_viterbi_algorithm(evaluation_data, token_set, transition_map, emission_map, label_set):
    """
    Evaluates the performance of the Viterbi algorithm on the evaluation dataset.

    Args:
        evaluation_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.
        token_set (set): A set of tokens seen during training.
        transition_map (dict): A dictionary mapping (label1, label2) pairs to their transition probabilities.
        emission_map (dict): A dictionary mapping (token, label) pairs to their emission probabilities.
        label_set (list): A list of all possible POS tags (labels).

    Returns:
        tuple: A tuple containing:
            - overall_error_rate (float): The error rate for all tokens.
            - known_error_rate (float): The error rate for tokens seen during training.
            - unknown_error_rate (float): The error rate for tokens not seen during training.
    """

    total_test_cases = 0
    correct_known, total_known = 0, 0
    correct_unknown, total_unknown = 0, 0

    for sentence in evaluation_data:
        viterbi_predictions = execute_viterbi_algorithm(transition_map, emission_map, label_set, sentence)
        for i in range(len(sentence)):
            token, actual_label = sentence[i]
            predicted_label = viterbi_predictions[i]
            if token in token_set:
                if actual_label == predicted_label:
                    correct_known += 1
                total_known += 1
            else:
                if 'NN' == actual_label:
                    correct_unknown += 1
                total_unknown += 1
            total_test_cases += 1

    overall_error_rate = 1 - ((correct_known + correct_unknown) / total_test_cases)
    known_error_rate = 1 - (correct_known / total_known)
    unknown_error_rate = 1 - (correct_unknown / total_unknown)
    return overall_error_rate, known_error_rate, unknown_error_rate


def evaluate_viterbi_with_smoothing(evaluation_data, vocabulary, transition_map, emission_map, label_set):
    """
    Evaluates the Viterbi algorithm's performance on the evaluation dataset with Add-One smoothing applied.

    Args:
        evaluation_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.
        vocabulary (set): A set of tokens seen during training.
        transition_map (dict): A dictionary mapping (label1, label2) pairs to their transition probabilities.
        emission_map (dict): A dictionary mapping (token, label) pairs to their emission probabilities.
        label_set (list): A list of all possible POS tags (labels).

    Returns:
        tuple: A tuple containing:
            - overall_error_rate (float): The error rate for all tokens.
            - known_error_rate (float): The error rate for tokens seen during training.
            - unknown_error_rate (float): The error rate for tokens not seen during training.
    """

    total_test_cases = 0
    correct_known, total_known = 0, 0
    correct_unknown, total_unknown = 0, 0

    for sentence in evaluation_data:
        viterbi_predictions = execute_viterbi_algorithm(transition_map, emission_map, label_set, sentence)
        for i in range(len(sentence)):
            token, actual_label = sentence[i]
            predicted_label = viterbi_predictions[i]
            if token in vocabulary:
                if actual_label == predicted_label:
                    correct_known += 1
                total_known += 1
            else:
                if actual_label == predicted_label:
                    correct_unknown += 1
                total_unknown += 1
            total_test_cases += 1

    overall_error_rate = 1 - ((correct_known + correct_unknown) / total_test_cases)
    known_error_rate = 1 - (correct_known / total_known)
    unknown_error_rate = 1 - (correct_unknown / total_unknown)
    return overall_error_rate, known_error_rate, unknown_error_rate


def evaluate_viterbi_with_pseudo_data(pseudo_evaluation_data, evaluation_data, vocabulary, transition_map, emission_map,
                                      label_set):
    """
    Evaluates the Viterbi algorithm's performance using pseudo-word data for evaluation.

    Args:
        pseudo_evaluation_data (list): A list of sentences, where rare tokens are replaced by pseudo-words.
        evaluation_data (list): The original evaluation dataset for comparison.
        vocabulary (set): A set of tokens seen during training.
        transition_map (dict): A dictionary mapping (label1, label2) pairs to their transition probabilities.
        emission_map (dict): A dictionary mapping (token, label) pairs to their emission probabilities.
        label_set (list): A list of all possible POS tags (labels).

    Returns:
        tuple: A tuple containing:
            - overall_error_rate (float): The error rate for all tokens.
            - known_error_rate (float): The error rate for tokens seen during training.
            - unknown_error_rate (float): The error rate for tokens not seen during training.
    """

    total_test_cases = 0
    correct_known, total_known = 0, 0
    correct_unknown, total_unknown = 0, 0

    for idx in range(len(pseudo_evaluation_data)):
        pseudo_sentence = pseudo_evaluation_data[idx]
        original_sentence = evaluation_data[idx]
        viterbi_predictions = execute_viterbi_algorithm(transition_map, emission_map, label_set, pseudo_sentence)
        for i in range(len(original_sentence)):
            token, actual_label = original_sentence[i]
            predicted_label = viterbi_predictions[i]
            if token in vocabulary:
                if actual_label == predicted_label:
                    correct_known += 1
                total_known += 1
            else:
                if actual_label == predicted_label:
                    correct_unknown += 1
                total_unknown += 1
            total_test_cases += 1

    overall_error_rate = 1 - ((correct_known + correct_unknown) / total_test_cases)
    known_error_rate = 1 - (correct_known / total_known)
    unknown_error_rate = 1 - (correct_unknown / total_unknown)
    return overall_error_rate, known_error_rate, unknown_error_rate


def create_pseudo_training_data(training_data):
    """
    Generates a pseudo-training dataset by replacing infrequent tokens with pseudo-word categories.

    Args:
        training_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.

    Returns:
        list: A pseudo-training dataset with rare tokens replaced by pseudo-word categories.
    """

    training_tokens = [sentence[i][0] for sentence in training_data for i in range(len(sentence))]
    token_frequencies = Counter(training_tokens)
    pseudo_training_data = []

    for sentence in training_data:
        modified_sentence = []
        for pair in sentence:
            token, label = pair
            if token_frequencies[token] < PSEUDO_WORD_THRESHOLD:
                token = classify_token(token)
            modified_sentence.append((token, label))
        pseudo_training_data.append(modified_sentence)

    return pseudo_training_data


def create_pseudo_evaluation_data(evaluation_data, full_vocabulary):
    """
    Generates a pseudo-evaluation dataset by replacing tokens not in the vocabulary with pseudo-word categories.

    Args:
        evaluation_data (list): A list of sentences, where each sentence is a list of (token, label) tuples.
        full_vocabulary (set): The full vocabulary of tokens seen during training.

    Returns:
        list: A pseudo-evaluation dataset with unknown tokens replaced by pseudo-word categories.
    """

    pseudo_evaluation_data = []

    for sentence in evaluation_data:
        modified_sentence = []
        for pair in sentence:
            token, label = pair
            if token not in full_vocabulary:
                token = classify_token(token)
            modified_sentence.append((token, label))
        pseudo_evaluation_data.append(modified_sentence)

    return pseudo_evaluation_data


def classify_token(token):
    """
    Classifies a token into a pseudo-word category based on its characteristics.

    Args:
        token (str): The token to be classified.

    Returns:
        str: The pseudo-word category assigned to the token.
    """

    if token.isdigit():
        return "<NUMERIC>"
    elif any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
        return "<ALNUM>"  # Alpha-numeric tokens
    elif any(char in token for char in '@#$%&'):
        return "<SPECIAL>"
    elif token.isupper():
        return "<ALLCAPS>"  # All-uppercase tokens
    elif token[0].isupper() and not token.isupper():
        return "<CAP>"
    elif token[-1] in ".,!?":
        return "<PUNCTUATED>"  # Tokens ending with punctuation
    elif len(set(token)) == 1 and len(token) > 2:
        return "<REPEATED>"
    elif len(token) <= 3:
        return "<SHORT>"  # Tokens of length 3 or less
    elif token.endswith(("ly", "able", "ible", "ment", "tion", "ness", "ship")):
        return "<SUFFIX>"
    elif "_" in token:
        return "<UNDERSCORE>"  # Tokens containing underscores
    elif token.startswith("http") or token.startswith("www") or token.endswith((".com", ".org", ".net")):
        return "<URL>"  # Tokens resembling URLs
    elif token.isnumeric() and len(token) == 4:
        return "<YEAR>"  # Four-digit numbers, likely years
    elif any(char in token for char in "[]{}()<>"):
        return "<BRACKETS>"
    elif any(char in token for char in "+-*/="):
        return "<OPERATOR>"
    elif "'" in token:
        return "<CONTRACTION>"  # Tokens containing apostrophes
    elif token.islower():
        return "<LOWERCASE>"
    elif len(token) > 12:
        return "<VERYLONG>"
    else:
        return "<RARE>"  # Default classification for unhandled tokens


def visualize_confusion_matrix(pseudo_evaluation_data, transition_map, emission_map, label_set):
    """
    Visualizes the confusion matrix for predictions made by the Viterbi algorithm in a compact form with truncation.

    Args:
        pseudo_evaluation_data (list): A list of sentences with pseudo-words, where each sentence is a list of (token, label) tuples.
        transition_map (dict): A dictionary mapping (label1, label2) pairs to their transition probabilities.
        emission_map (dict): A dictionary mapping (token, label) pairs to their emission probabilities.
        label_set (list): A list of all possible POS tags (labels).

    Prints:
        - A compact confusion matrix showing the start, end, and truncated rows and columns.
        - The dimensions of the matrix.
        - The 10 most frequent errors.
    """
    from tabulate import tabulate  # For a clean tabular display

    true_labels = []
    predicted_labels = []

    # Generate true and predicted labels
    for sentence in pseudo_evaluation_data:
        sentence_true_labels = [token_label[1] for token_label in sentence]
        predicted_sequence = execute_viterbi_algorithm(transition_map, emission_map, label_set, sentence)
        true_labels.extend(sentence_true_labels)
        predicted_labels.extend(predicted_sequence)

    # Create confusion matrix
    unique_labels = sorted(set(true_labels + predicted_labels))
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

    # Define truncation limit for rows and columns
    truncation_limit = 5  # Number of rows/columns to show at the start and end
    total_labels = len(unique_labels)

    # Helper function to format a row with truncation
    def format_row(label_idx):
        row = [unique_labels[label_idx]]
        row += list(conf_matrix[label_idx][:truncation_limit])
        if total_labels > 2 * truncation_limit:
            row.append("...")
        row += list(conf_matrix[label_idx][-truncation_limit:])
        return row

    # Prepare truncated rows
    truncated_rows = []
    for i in range(truncation_limit):
        truncated_rows.append(format_row(i))
    if total_labels > 2 * truncation_limit:
        truncated_rows.append(["..."] + ["..."] * (truncation_limit + 1))
    for i in range(total_labels - truncation_limit, total_labels):
        truncated_rows.append(format_row(i))

    # Prepare headers with truncation
    truncated_headers = [" "] + unique_labels[:truncation_limit]
    if total_labels > 2 * truncation_limit:
        truncated_headers.append("...")
    truncated_headers += unique_labels[-truncation_limit:]

    # Print truncated confusion matrix
    print("\n--- Truncated Confusion Matrix ---")
    print(tabulate(truncated_rows, headers=truncated_headers, tablefmt="plain"))

    # Print matrix dimensions
    print(f"\n[{total_labels} rows x {total_labels} columns]")

    # Find the 10 most frequent errors
    errors = []
    for i, true_label in enumerate(unique_labels):
        for j, predicted_label in enumerate(unique_labels):
            if i != j:  # Exclude diagonal (correct classifications)
                errors.append((true_label, predicted_label, conf_matrix[i, j]))

    # Sort errors by frequency (descending) and take the top 10
    most_frequent_errors = sorted(errors, key=lambda x: x[2], reverse=True)[:10]

    print("\n--- 10 Most Frequent Errors ---")
    print(tabulate(most_frequent_errors, headers=["True Label", "Predicted Label", "Count"], tablefmt="plain"))



if __name__ == '__main__':
    print("\n--- Task A: Data Preparation ---")
    training_data, evaluation_data = process_and_divide_dataset()

    # Calculate dataset sizes
    total_sentences = len(training_data) + len(evaluation_data)
    training_size = len(training_data)
    test_size = len(evaluation_data)

    # Print the dataset information
    print(f"Total sentences in 'news' category: {total_sentences}")
    print(f"Training set size: {training_size} sentences")
    print(f"Test set size: {test_size} sentences")

    print("\n--- Task B: Maximum Likelihood Estimation (MLE) ---")
    mle_map, tag_statistics = compute_mle(training_data)
    mle_error_metrics = test_mle_error_rates(mle_map, evaluation_data)
    print(f'---- Analysis B ----')
    print(f'Total Error Rate = {mle_error_metrics[0] * 100:.2f}%')
    print(f'Known Words Error Rate = {mle_error_metrics[1] * 100:.2f}%')
    print(f'Unknown Words Error Rate = {mle_error_metrics[2] * 100:.2f}%')

    print("\n--- Task C: Hidden Markov Model (HMM) without Smoothing ---")
    label_set = set([token_label[1] for sent in training_data for token_label in sent])
    vocabulary = set([token_label[0] for sent in training_data for token_label in sent])
    apply_smoothing = False
    emission_map, transition_map = train_hidden_markov_model(training_data, evaluation_data, apply_smoothing)
    hmm_error_metrics = evaluate_viterbi_algorithm(evaluation_data, vocabulary, transition_map, emission_map, label_set)
    print(f'---- Analysis C ----')
    print(f'Total Error Rate = {hmm_error_metrics[0] * 100:.2f}%')
    print(f'Known Words Error Rate = {hmm_error_metrics[1] * 100:.2f}%')
    print(f'Unknown Words Error Rate = {hmm_error_metrics[2] * 100:.2f}%')

    print("\n--- Task D: Hidden Markov Model (HMM) with Smoothing ---")
    apply_smoothing = True
    emission_map, transition_map = train_hidden_markov_model(training_data, evaluation_data, apply_smoothing)
    smoothed_error_metrics = evaluate_viterbi_with_smoothing(evaluation_data, vocabulary, transition_map, emission_map, label_set)
    print(f'---- Analysis D ----')
    print(f'Total Error Rate = {smoothed_error_metrics[0] * 100:.2f}%')
    print(f'Known Words Error Rate = {smoothed_error_metrics[1] * 100:.2f}%')
    print(f'Unknown Words Error Rate = {smoothed_error_metrics[2] * 100:.2f}%')

    print("\n--- Task E: Pseudo-Word Analysis ---")
    pseudo_training_data = create_pseudo_training_data(training_data)
    extended_vocabulary = set([token_label[0] for sent in pseudo_training_data for token_label in sent])
    pseudo_evaluation_data = create_pseudo_evaluation_data(evaluation_data, extended_vocabulary)

    emission_map, transition_map = train_hidden_markov_model(pseudo_training_data, pseudo_evaluation_data, apply_smoothing=False)
    pseudo_error_metrics = evaluate_viterbi_with_pseudo_data(pseudo_evaluation_data, evaluation_data, vocabulary, transition_map, emission_map, label_set)
    print(f'---- Analysis E ( II) ----')
    print(f'Total Error Rate = {pseudo_error_metrics[0] * 100:.2f}%')
    print(f'Known Words Error Rate = {pseudo_error_metrics[1] * 100:.2f}%')
    print(f'Unknown Words Error Rate = {pseudo_error_metrics[2] * 100:.2f}%')

    emission_map, transition_map = train_hidden_markov_model(pseudo_training_data, pseudo_evaluation_data, apply_smoothing=True)
    smoothed_pseudo_error_metrics = evaluate_viterbi_with_pseudo_data(pseudo_evaluation_data, evaluation_data, vocabulary, transition_map, emission_map, label_set)
    print(f'---- Analysis E (III) ----')
    print(f'Total Error Rate = {smoothed_pseudo_error_metrics[0] * 100:.2f}%')
    print(f'Known Words Error Rate = {smoothed_pseudo_error_metrics[1] * 100:.2f}%')
    print(f'Unknown Words Error Rate = {smoothed_pseudo_error_metrics[2] * 100:.2f}%')

    print("\n--- Confusion Matrix Visualization ---")
    visualize_confusion_matrix(pseudo_evaluation_data, transition_map, emission_map, label_set)
