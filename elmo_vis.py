from collections import OrderedDict

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.decomposition import PCA


class Elmo:
    def __init__(self):
        self.elmo = ElmoEmbedder()

    def get_elmo_vector(self, tokens, layer):
        vectors = self.elmo.embed_sentence(tokens)
        X = []
        for vector in vectors[layer]:
            X.append(vector)

        X = np.array(X)

        return X


def dim_reduction(X, n):
    pca = PCA(n_components=n)
    print("size of X: {}".format(X.shape))
    results = pca.fit_transform(X)
    print("size of reduced X: {}".format(results.shape))

    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print("Variance retained ratio of PCA-{}: {}".format(i+1, ratio))

    return results


def plot(word, token_list, reduced_X, file_name, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # plot ELMo vectors
    i = 0
    for j, token in enumerate(token_list):
        color = pick_color(j)
        for _, w in enumerate(token):

            # only plot the word of interest
            if w.lower() in [word, word + 's', word + 'ing', word + 'ed']:
                ax.plot(reduced_X[i, 0], reduced_X[i, 1], color)
            i += 1

    tokens = []
    for token in token_list:
        tokens += token

    # annotate point
    k = 0
    for i, token in enumerate(tokens):
        if token.lower() in [word, word + 's', word + 'ing', word + 'ed']:
            text = ' '.join(token_list[k])

            # bold the word of interest in the sentence
            text = text.replace(token, r"$\bf{" + token + "}$")

            plt.annotate(text, xy=(reduced_X[i, 0], reduced_X[i, 1]))
            k += 1

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.savefig(file_name, bbox_inches="tight")

    print("{} saved\n".format(file_name))


def pick_color(i):
    if i == 0:
        color = 'ro'
    elif i == 1:
        color = 'bo'
    elif i == 2:
        color = 'yo'
    elif i == 3:
        color = 'go'
    else:
        color = 'co'
    return color


if __name__ == "__main__":
    model = Elmo()

    banks = OrderedDict()
    banks[0] = "One can deposit money at the bank"
    banks[1] = "He had a nice walk along the river bank"
    banks[2] = "I withdrew cash from the bank"
    banks[3] = "The river bank was not clean"
    banks[4] = "My wife and I have a joint bank account"

    works = OrderedDict()
    works[0] = "I like this beautiful work by Andy Warhol"
    works[1] = "Employee works hard every day"
    works[2] = "My sister works at Starbucks"
    works[3] = "This amazing work was done in the early nineteenth century"
    works[4] = "Hundreds of people work in this building"

    plants = OrderedDict()
    plants[0] = "The gardener planted some trees in my yard"
    plants[1] = "I plan to plant a Joshua tree tomorrow"
    plants[2] = "My sister planted a seed and hopes it will grow to a tree"
    plants[3] = "This kind of plant only grows in the subtropical region"
    plants[4] = "Most of the plants will die without water"

    words = {
        "bank": banks,
        "work": works,
        "plant": plants
    }

    # contextual vectors for ELMo layer 1 and 2
    for layer in [1, 2]:
        for word, sentences in words.items():
            print("visualizing word {} using ELMo layer {}".format(word, layer))
            X = np.concatenate([model.get_elmo_vector(tokens=sentences[idx].split(),
                                                      layer=layer)
                                for idx, _ in enumerate(sentences)], axis=0)

            # The first 2 principal components
            X_reduce = dim_reduction(X=X, n=2)

            token_list = []
            for _, sentence in sentences.items():
                token_list.append(sentence.split())

            file_name = "{}_elmo_layer_{}.png".format(word, layer)
            title = "Layer {} ELMo vectors of the word {}".format(layer, word)
            plot(word, token_list, X_reduce, file_name, title)

