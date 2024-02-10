import tweepy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Set up Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Function to clean and preprocess tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower().strip()
    return tweet

# Function to fetch tweets based on a query
def fetch_tweets(query, num_tweets=100):
    tweets = []
    for tweet in tweepy.Cursor(api.search, q=query, lang='en', tweet_mode='extended').items(num_tweets):
        tweets.append(tweet.full_text)
    return tweets

# Fetch positive and negative tweets (you may need to manually label a dataset for this)
positive_tweets = fetch_tweets('happy', num_tweets=100)
negative_tweets = fetch_tweets('sad', num_tweets=100)

# Create a labeled dataset
positive_labels = [1] * len(positive_tweets)
negative_labels = [0] * len(negative_tweets)
all_tweets = positive_tweets + negative_tweets
all_labels = positive_labels + negative_labels

# Preprocess tweets
all_tweets = [preprocess_tweet(tweet) for tweet in all_tweets]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(all_tweets, all_labels, test_size=0.2, random_state=42)

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopwords.words('english'))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
