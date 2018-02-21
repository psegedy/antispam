# antispam
Spam filter demonstration using machine learning

Classify emails in .eml format if it is SPAM or HAM using [scikit-learn](http://scikit-learn.org) python module. Before first run it is neccessary to have training dataset for the classifier. Simply change `DATASET` list with path to training data and values `True` for SPAM and `False` for HAM. For this demonstration I'm using some emails from this pre-classified dataset http://www2.aueb.gr/users/ion/data/enron-spam/

Example of usage
```bash
./antispam.py email.eml email2.eml email3.eml email4.eml
```

Output
```bash
email.eml - OK 
email2.eml - SPAM
email3.eml - SPAM 
email4.eml - FAIL
```
OK - it's HAM
SPAM - it's SPAM
FAIL - failed to load or classify email

Note: Once classifier is trained, you can comment call to `train_data()` method. It is project for BIS class @[FIT VUT](http://www.fit.vutbr.cz) and it was submitted with trained classifier.
