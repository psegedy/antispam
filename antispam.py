#!/usr/bin/env python3

# Author: Patrik Segedy <xseged00@vutbr.cz>
# File: detector.py
# Description: Simple antispam using machine learning for BIS class

import os
import sys
import email.parser
import email.message
import email.policy
import html2text
from pandas import DataFrame
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


DEFAULT_CHARSET = 'latin-1'
# http://www2.aueb.gr/users/ion/data/enron-spam/
# + private e-mails
DATASET = [
    ('ham/beck-s',      False),
    ('ham/kaminski-v',  False),
    ('priklady/emaily', False),
    ('ham/Inbox_20171208-1919', False),
    ('ham/farmer-d',    False),
    # ('ham/kitchen-l',   False),
    # ('ham/lokay-m',     False),
    # ('ham/williams-w3', False),
    # ('spam/GP',          True),
    # ('spam/BG',          True),
    ('spam/SH',          True),
    ('priklady/spamy',   True)
]


def read_dataset(path):
    """Read dataset and yield file path and content of email body"""
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = (os.path.join(root, file))
            with open(file_path, 'rb') as f:
                try:
                    msg = email.message_from_binary_file(f, policy=email.policy.default)
                    try:
                        msg_body = msg.get_body(preferencelist=('html', 'plain'))
                        msg_body_payload = msg_body.get_payload(decode=True)
                        charset = msg_body.get_content_charset(DEFAULT_CHARSET)
                    except:
                        msg_body_payload = msg.get_payload(decode=True)
                        charset = msg.get_content_charset(DEFAULT_CHARSET)
                    # try to decode payload
                    try:
                        msg_payload_decoded = msg_body_payload.decode(charset)
                    except:
                        charset = DEFAULT_CHARSET
                        msg_payload_decoded = msg_body_payload.decode(charset)

                    lines = [html2text.html2text(line).strip() for line in msg_payload_decoded.splitlines() if html2text.html2text(line).strip()]
                except:
                    lines = []

            yield file_path, "".join(lines)


def make_data_frame(path, is_spam):
    """Construct DataFrame from email body and class (spam/ham)"""
    rows = []
    index = []
    for file, content in read_dataset(path):
        rows.append({'content': content, 'class': is_spam})
        index.append(file)

    data_frame = DataFrame(rows, index=index)
    return data_frame


def train_model():
    # add data to dataframe
    data = DataFrame({'content': [], 'class': []})
    for path, is_spam in DATASET:
        data = data.append(make_data_frame(path, is_spam))

    # shuffle dataset
    data = data.reindex(numpy.random.permutation(data.index))

    # count number of words, frequentions, and classify
    pipeline = Pipeline([
        ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2))),
        ('tfidf_transformer',  TfidfTransformer()),
        ('classifier',         MultinomialNB())
    ])
    # fit data
    pipeline.fit(data['content'].values, data['class'].values)

    # dump classifier
    joblib.dump(pipeline, 'trained_data.pkl', compress=9)

    return pipeline


def classify(model, emails):
    """Classify given email body"""
    return model.predict(emails)


def read_argv():
    """Read emails from argv, and yield file name and email body"""
    for arg in sys.argv[1:]:
        try:
            with open(arg, 'rb') as f:
                msg = email.message_from_binary_file(f, policy=email.policy.default)
                try:
                    msg_body = msg.get_body(preferencelist=('html', 'plain'))
                    msg_body_payload = msg_body.get_payload(decode=True)
                    charset = msg_body.get_content_charset(DEFAULT_CHARSET)
                except:
                    msg_body_payload = msg.get_payload(decode=True)
                    charset = msg.get_content_charset(DEFAULT_CHARSET)
                # try to decode payload
                try:
                    msg_payload_decoded = msg_body_payload.decode(charset)
                except:
                    charset = DEFAULT_CHARSET
                    msg_payload_decoded = msg_body_payload.decode(charset)

                lines = [html2text.html2text(line).strip() for line in msg_payload_decoded.splitlines() if html2text.html2text(line).strip()]
        except:
            print(arg, '- FAIL')
            continue

        yield arg, "".join(lines)

if __name__ == '__main__':
    train_model()
    # load classifier
    model = joblib.load('trained_data.pkl')
    for file, content in read_argv():
        prediction = classify(model, [content])
        if prediction[0] == 1.0:
            print(file, '- SPAM')
        elif prediction[0] == 0.0:
            print(file, '- OK')
