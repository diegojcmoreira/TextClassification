import gensim
import os
import pandas as pd

from DataPreProcess import build_csv_with_data
from DataPreProcess import read_data

from Embedding import bag_of_word_embedding
from Embedding import tfidf_embedding

from Classification import predict_evaluate_cv

from sklearn.svm import SVC

def load_word2vec_model_from_file(filepath: str) -> object:
    return gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(filepath)

if __name__ == "__main__":
    working_folder = os.getcwd()
    dataset_folder = os.path.join(working_folder, os.path.join("Dataset","Escopo"))



    print(dataset_folder)

    filepath_data_csv = build_csv_with_data(working_folder, dataset_folder,"escopo.csv")

    data_frame_escopo = read_data(filepath_data_csv)

    #BOW_embedding
    print("BOW Embedding")
    embedding_bow = bag_of_word_embedding(data_frame_escopo)
    predict_evaluate_cv(embedding_bow, data_frame_escopo['Class'].values.ravel(), SVC(kernel='linear', probability=True))

    print("TFIDF Embedding")
    embedding_tfidf = tfidf_embedding(data_frame_escopo)
    predict_evaluate_cv(embedding_tfidf, data_frame_escopo['Class'].values.ravel(), SVC(kernel='linear', probability=True))

    #
    #
    #
    #
    #
    # model_filename = "cbow_s1000.txt"
    #
    #
    # model_filepath = os.path.join(working_folder, os.path.join("Word2Vec",model_filename))
    #
    # w2v_model = load_word2vec_model_from_file(model_filepath)
    #
    # test_word = "elevador"
    #
    # print("Palavra mais similar a {0}".format(test_word), w2v_model.wv.most_similar(positive=test_word))
    #
    #
    #
    # print(model_filepath)

