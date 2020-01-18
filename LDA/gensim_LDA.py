import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel  # Compute Coherence Score
import pandas as pd
import numpy as np
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import sys


NUM_TOPICS = 5

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def create_bow(data):
    # Create Dictionary
    word_dict = corpora.Dictionary(data)  # Create Corpus
    texts = data
    # Term Document Frequency
    corpus = [word_dict.doc2bow(text) for text in texts]  # View
    return corpus, word_dict


# supporting function
def compute_coherence_values(corpus, text_data, dictionary, k, a, b):
    """
        Computer the c_v coherence score for an arbitrary LDA model.

        :param corpus: the text to be modelled (a list of vectors).
        :param text_data: the actual text as a list of list
        :param dictionary: a dictionary coresponding that maps elements of the corpus to words.
        :param k: the number of topics
        :param a: Alpha, document-topic density
        :param b: Beta, topic-word density
    """

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           per_word_topics=True)

    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=text_data, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()


def hyper_parameter_tuning(corpus, word_dict, text_data):
    min_topics = 5
    max_topics = 11
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric') #REMOVE
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')  # Validation sets
    model_results = {'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }  # Can take a long time to run
    num_combinations = len(topics_range)*len(alpha)*len(beta)
    pbar = tqdm.tqdm(total=num_combinations)
    # iterate through number of topics, different alpha values, and different beta values
    for k in topics_range:
        for a in alpha:
            for b in beta:
                # get the coherence score for the given parameters
                cv = compute_coherence_values(
                    corpus=corpus, text_data=text_data, dictionary=word_dict, k=k, a=a, b=b)
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)
                pbar.update(1)

    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    best_val = np.argmax(model_results["Coherence"])
    pbar.close()
    print("Best c_v val: {} (alpha: {}, beta: {}, topics: {})".format(model_results['Coherence'][best_val],model_results['Alpha'][best_val], model_results['Beta'][best_val],model_results['Topics'][best_val]))
    return model_results['Coherence'][best_val],model_results['Alpha'][best_val], model_results['Beta'][best_val],model_results['Topics'][best_val]

def vis_coherence_surface(file_path,topics=10):
    data = pd.read_csv(file_path)
    data = data[data["Topics"]==10]
    x = data["Alpha"].apply(lambda x : 0.1 if x=="symmetric" or x=="asymmetric" else x).astype('float64')
    y = data["Beta"].apply(lambda x : 0.1 if x=="symmetric" or x=="asymmetric" else x).astype('float64')
    z = data["Coherence"].astype('float64')
    
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Coherence (c_v)')
    plt.title("Alpha-Beta Hyperparameter Sweep (k={})".format(topics))
    plt.savefig('Coherence_Surface_k={}.png'.format(topics))
    plt.show()


if __name__ == "__main__":
    vis_coherence_surface("good_lda_tuning_results.csv",9)
# if __name__ == "__main__":
#     # Put all of the party leaders into one data frame
#     usernames = sys.argv[1:]
#     frames = []
#     for username in usernames:
#         file_path = "../data/{}_data.csv".format(username)
#         timeline_df = pd.read_csv(file_path)
#         print("Number of Tweets for {} is {}".format(username, len(timeline_df)))
#         frames.append(timeline_df)
#     text_data = pd.concat(frames)["clean_text"].values.astype('U')
#     text_data = [sent.split() for sent in text_data]
#     # Build the bigram models
#     print("--- finding bigrams ---")
#     bigram = gensim.models.Phrases(text_data, min_count=5, threshold=100)
#     bigram_mod = gensim.models.phrases.Phraser(bigram)
#     # creates bigrams of words that appear frequently together "gun control" -> "gun_control"
#     text_data = make_bigrams(text_data, bigram_mod)
#     print("--- creating BoW model ---")
#     corpus, word_dict = create_bow(text_data)
#     print("--- starting hyperparameter tuning ---")
#     coherence,alpha,beta,num_topics = hyper_parameter_tuning(corpus, word_dict, text_data)
#     # Build LDA model
#     lda_model = gensim.models.LdaMulticore(corpus=corpus,
#                                            id2word=word_dict,
#                                            num_topics=num_topics,
#                                            alpha=alpha,
#                                            eta=beta,
#                                            random_state=100,
#                                            chunksize=100,
#                                            passes=10,
#                                            per_word_topics=True)

#     for idx, topic in lda_model.print_topics(-1):
#         print('Topic: {} \nWords: {}'.format(idx, topic))

#     coherence_model_lda = CoherenceModel(
#         model=lda_model, texts=text_data, dictionary=word_dict, coherence='c_v')
#     coherence_lda = coherence_model_lda.get_coherence()
#     print('\nCoherence Score: ', coherence_lda)
#     lda_model.save("gensim_lda")

