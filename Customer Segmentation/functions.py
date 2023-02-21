import requests
import time
import json
import re
import contractions
import string
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import pandas as pd
from datetime import datetime
from collections import defaultdict


def remove_unwanted(post):
    result = re.sub(r"\n", " ", post)
    result = re.sub("&amp;#x200B;", " ", result)
    pattern = re.compile(
        r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-;]*[\w@?^=%&\/~+#-;]*)"
    )
    result = re.sub(pattern, " ", result)
    pattern = re.compile(r"([\w_-]+\.com)")
    result = re.sub(pattern, " ", result)
    return result


def retrieve_posts(subreddit, min_word_count=20, no_of_posts=100):
    with open(f"{subreddit}.json", "w") as f:
        counter = 0
        current_time = datetime.now().timestamp()
        all_posts = []
        num_posts_retrieved = no_of_posts if no_of_posts < 1000 else 1000
        while counter < no_of_posts:
            try:
                response = requests.get(
                    f"https://api.pushshift.io/reddit/search/submission?subreddit={subreddit}&is_Video=false&size={str(num_posts_retrieved)}&sort=desc&before={str(int(current_time))}"
                )
                total_posts = response.json()["data"]
                for post in total_posts:
                    if ("title" and "selftext") in post.keys():
                        contents = remove_unwanted(
                            post["title"] + " " + post["selftext"]
                        )
                        if len(contents.split()) > min_word_count:
                            all_posts.append(dict(content=contents))
                            print(post["full_link"])
                current_time = total_posts[-1]["created_utc"]
                counter = len(all_posts)
                print(
                    f"Retrieved {counter} posts out of {no_of_posts} posts required ({(counter / no_of_posts) * 100}%)."
                )
                print(
                    f"Date and time of last post from this API call: {str(datetime.fromtimestamp(current_time))}"
                )
                print("Sleeping....")
                time.sleep(5)
                print("Awake! Let's continue!")
            except Exception as e:
                print(e)
        json.dump(all_posts, f, indent=4)
        print("Done!")


def tokenize_and_extract_words(description):
    content_words = []
    for post in subreddit:
        no_contraction = [
            contractions.fix(word) for word in post["content"].split()
        ]
        tokenized = word_tokenize(" ".join(no_contraction))
        lower_not_punctuation = [
            word.lower() for word in tokenized if word not in string.punctuation
        ]
        result = re.findall(r"[a-zA-Z]{1,}", " ".join(lower_not_punctuation))
        post["words"] = result
        content_words.append(post)

    f.seek(0)
    json.dump(content_words, f, indent=4)


def text_counts(word_column):
    eda_counts = pd.DataFrame()
    eda_counts["word_count"] = word_column.apply(len)
    eda_counts["char_count"] = word_column.apply(lambda x: len("".join(x)))
    eda_counts["word_density"] = eda_counts["word_count"] / (
        eda_counts["char_count"] + 1
    )
    return eda_counts


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def metrics(model, X_train, y_train, X_cv, y_cv):
    val = defaultdict(dict)
    val[str(model)]["train_accuracy"] = model.score(X_train, y_train) 
    val[str(model)]["validation_accuracy"] = cross_val_score(
        model, X_train, y_train, cv=5
    ).mean()
    val[str(model)]["train_precision"] = precision_score(
        y_train, model.predict(X_train)
    )
    val[str(model)]["validation_precision"] = precision_score(y_cv, model.predict(X_cv))
    return val
