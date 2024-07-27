import os
import subprocess
import re

def generate_ngram_model(corpus_file, model_file, order=9):
    srilm_bin_dir = r'D:/自然语言/SRILM/For PC'  
    ngram_count_cmd = os.path.join(srilm_bin_dir, 'ngram-count')
    cmd = [
        ngram_count_cmd,
        '-text', corpus_file,
        '-order', str(order),
        '-lm', model_file,
        '-kndiscount',
        '-interpolate'
    ]
    subprocess.run(cmd, check=True)


def query_srilm_prob(ngram, srilm_model_path):
    srilm_bin_dir = r'D:/自然语言/SRILM/For PC' 
    ngram_cmd = os.path.join(srilm_bin_dir, 'ngram')
    cmd = f'echo "{ " ".join(ngram) }" | {ngram_cmd} -lm {srilm_model_path} -ppl - -debug 2'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    match = re.search(r'logprob= (-?\d+\.\d+)', result.stdout)
    if match:
        return float(match.group(1))
    return -float('inf')


def laplace_smoothing(ngram_counts, vocab_size):
    smoothed_probs = {}
    total_count = sum(ngram_counts.values())
    for ngram, count in ngram_counts.items():
        smoothed_count = count + 1
        smoothed_probs[ngram] = smoothed_count / (total_count + vocab_size)
    return smoothed_probs

def get_context(word, sentence, n):
    tokens = sentence.split()
    word_index = tokens.index(word)
    start = max(0, word_index - n)
    return tokens[start:word_index]

def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    return dp[m][n]

def generate_candidates(word, vocab, max_distance):
    candidates = set()
    for vocab_word in vocab:
        if edit_distance(word, vocab_word) <= max_distance:
            candidates.add(vocab_word)
    return candidates

def read_vocab(filename):
    vocab = set()
    with open(filename, 'r') as file:
        for line in file:
            word = line.strip()
            vocab.add(word)
    return vocab

def spell_correct_word(word, sentence, candidates, vocab, srilm_model_path, max_n):
    max_prob = -float('inf')
    best_candidate = word
    for candidate in candidates:
        prob = 1.0 / len(vocab)  # Default probability
        for n in range(1, max_n + 1):
            context = get_context(word, sentence, n)
            if len(context) >= n:  # Ensure enough context words are available
                gram = tuple(context[-(n-1):]) + (candidate,)
                prob_n = query_srilm_prob(gram, srilm_model_path)
                if prob_n > 0:
                    prob = prob_n
                    break
        if prob > max_prob:
            max_prob = prob
            best_candidate = candidate
    
    return best_candidate


def spell_correct_sentence(sentence, vocab, srilm_model_path, max_edit_distance, max_n):
    words = sentence.split()
    corrected_words = []
    for word in words:
        if word not in vocab:
            candidates = generate_candidates(word, vocab, max_edit_distance)
            word = spell_correct_word(word, sentence, candidates, vocab, srilm_model_path, max_n)
        corrected_words.append(word)
    return ' '.join(corrected_words)

def process_testdata(testdata_file, result_file, vocab_file, srilm_model_path, max_edit_distance, max_n):
    vocab = read_vocab(vocab_file)
    with open(testdata_file, 'r') as tf, open(result_file, 'w') as rf:
        for line in tf:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue  # Skip lines with unexpected format
            sentence_id = parts[0]
            sentence = parts[2]
            corrected_sentence = spell_correct_sentence(sentence, vocab, srilm_model_path, max_edit_distance, max_n)
            rf.write(f"{sentence_id}\t{corrected_sentence}\n")

# 示例文件路径（请确保这些文件路径和文件名正确）
testdata_file = './testdata.txt'
result_file = './result.txt'
vocab_file = './vocab.txt'
corpus_file = './corpus.txt'
srilm_model_path = './model.lm'

# 生成n-gram模型
generate_ngram_model(corpus_file, srilm_model_path)

# 处理测试数据并进行拼写纠错
process_testdata(testdata_file, result_file, vocab_file, srilm_model_path, 1, 9)
