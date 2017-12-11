import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.pipeline
from textblob import TextBlob

data = pandas.read_csv('labeled_messages.csv', sep=',')  # loads CSV into pandas dataframe

data['message'] = data.message.astype(str)  # converts messages within dataframe to string
data['length'] = data['message'].map(lambda text: len(text))  # adds column to dataframe showing length of each message

data.hist(column='length', by='is_spam', bins=50,
          range=[0, 250])  # creates histogram showing length of spam vs not spam messages


def split_into_tokens(message):  # tokenizes each message
    return TextBlob(message).words


def split_into_lemmas(message):  # take the lemma (base form) of each word
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]


bow_transformer = sklearn.feature_extraction.text.CountVectorizer(analyzer=split_into_lemmas).fit(data['message'])
# converts list of tokens from messages into a vector

messages_bow = bow_transformer.transform(data['message'])  # transforms all messages into a matrix

tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer().fit(
    messages_bow)  # use TF-IDF to adjust weights and normalize for each lemma

messages_tfidf = tfidf_transformer.transform(messages_bow)  # TF-IDF weighting and normalization on all messages

spam_detector = sklearn.naive_bayes.MultinomialNB().fit(messages_tfidf, data['is_spam'])  # creates NB classifier
all_predictions = spam_detector.predict(messages_tfidf)  # predicts if spam for every message

plt.matshow(sklearn.metrics.confusion_matrix(data['is_spam'], all_predictions), cmap=plt.cm.binary,
            interpolation='nearest')  # plots confusion matrix
plt.title('Naive Bayes Confusion Matrix')
plt.colorbar()
plt.ylabel('Expected Value')
plt.xlabel('Predicted Value')

msg_train, msg_test, label_train, label_test = sklearn.model_selection.train_test_split(data['message'],
                                                                                        data['is_spam'],
                                                                                        test_size=0.2)
                                                                                        # splits test and training data

pipeline = sklearn.pipeline.Pipeline([  # creates pipeline for naive bayes model
    ('bow', sklearn.feature_extraction.text.CountVectorizer(analyzer=split_into_lemmas)),
    # strings to token integer counts
    ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', sklearn.naive_bayes.MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

scores = sklearn.model_selection.cross_val_score(pipeline,  # steps to convert raw messages into models
                                                 msg_train,  # training data
                                                 label_train,  # training labels
                                                 cv=10,
                                                 # split data randomly into 10 parts: 9 for training, 1 for scoring
                                                 scoring='accuracy',  # score using accuracy
                                                 n_jobs=-1,
                                                 )
print('scores', scores)

def plot_learning_curve(filename, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    # creates plot of training and test accuracy
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = sklearn.model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    plt.savefig(filename, format='png')
    return plt

print('accuracy', sklearn.metrics.accuracy_score(data['is_spam'], all_predictions))
print('confusion matrix\n', sklearn.metrics.confusion_matrix(data['is_spam'], all_predictions))
plot_learning_curve('nbaccuracy.png', pipeline, "Naive Bayes Accuracy vs. Training Set Size", msg_train, label_train, cv=10)

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = sklearn.model_selection.GridSearchCV(
    pipeline,  # use pipeline created above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best parameter combination
    n_jobs=-1,
    scoring='accuracy',  # optimize for best accuracy
    cv=sklearn.model_selection.StratifiedKFold(n_splits=5),  # cross-validation
)

nb_detector = grid.fit(msg_train, label_train)

predictions = nb_detector.predict(
    msg_test)  # shows best parameter combinations, confirms that we should use idf and split_into_lemmas on our data

pipeline_svm = sklearn.pipeline.Pipeline([  # creates pipeline for svm model
    ('bow', sklearn.feature_extraction.text.CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
    ('classifier', sklearn.svm.SVC()),
])

param_svm = [  # automatically explores and tunes model using given pipeline parameters
    {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__kernel': ['linear']},
    {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = sklearn.model_selection.GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=sklearn.model_selection.StratifiedKFold(n_splits=5),  # what type of cross validation to use
)
svm_detector = grid_svm.fit(msg_train, label_train)  # find the best combination from param_svm

plot_learning_curve('svmaccuracy.png', pipeline_svm, "svm accuracy vs. training set size", msg_train, label_train, cv=5)
