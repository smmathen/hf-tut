from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
messages = [
    "I am so excited to watch Tyrese Haliburton play tonight!",
    "I was not impressed with Giannias Antetokounmpo's reaction after the game last night"
]

# scores = classifier(messages)
# print(scores)

# Zero Shot Classification
classifer = pipeline("zero-shot-classification")
messages = [
    "I am so excited to watch Tyrese Haliburton play tonight!",
]

candidate_labels = ["sports", "politics", "education"]

# scores = classifer(messages, candidate_labels=candidate_labels)
# print(scores)


# Text Generation
generator = pipeline("text-generation", model="distilgpt2")
messages = [
    "I want to see the Indianapolis Colts",
]

output = generator(messages, max_length=100, num_return_sequences=2)
print(output)
