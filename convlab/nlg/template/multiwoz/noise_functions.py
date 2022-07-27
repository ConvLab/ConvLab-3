# Valentin Mace
# valentin.mace@kedgebs.com
# Developed at Qwant Research

"""Functions adding noise to text"""

import random
import string


def spelling_noise(phrase, prob=0.1):
    new_phrase = []
    words = phrase.split(' ')
    for word in words:
        outcome = random.random()
        if outcome <= prob and word:
            ix = random.choice(range(len(word)))
            new_word = ''.join([word[w] if w != ix else random.choice(string.ascii_letters) for w in range(len(word))])
            new_phrase.append(new_word)
        else:
            new_phrase.append(word)

    return ' '.join([w for w in new_phrase])


def random_bool(probability=0.5):
    """Returns True with given probability
    Args:
        probability: probability to return True
    """
    assert (0 <= probability <= 1), "probability needs to be >= 0 and <= 1"
    return random.random() < probability


def delete_random_token(line, probability=0.1):
    """Delete random tokens in a given String with given probability
    Args:
        line: a String
        probability: probability to delete each token
    """
    line_split = line.split()
    ret = [token for token in line_split if not random_bool(probability)]
    return " ".join(ret)


def random_token_permutation(line, _range=3, probability=0.1):
    """Random permutation over the tokens of a String, restricted to a range, drawn from the uniform distribution
    Args:
        line: a String
        _range: Max range for token permutation
    """
    if random_bool(probability):
        line_split = line.split()
        new_indices = [i+random.uniform(0, _range+1) for i in range(len(line_split))]
        res = [x for _, x in sorted(zip(new_indices, line_split), key=lambda pair: pair[0])]
        return " ".join(res)
    return line
