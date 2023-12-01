import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import certifi
import ssl
import pickle
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split


# Set an unverified context for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
# Without this an error is shown that you need to download the package 'punkt'
nltk.download('punkt')

# This is eventually used to stem words
stemmer = LancasterStemmer()

torch.manual_seed(42)

# Loading the intents file as a json format
with open("intents.json") as file:
    data = json.load(file)
try:

    # opening the data.pickle file with the 'read bytes' property
    with open("data.pickle", "rb") as f:
        #only lists we need for our model
        words, labels, training, output = pickle.load(f)
except:
    words = [] # all the words inside a pattern
    labels = [] # the tag of the intent
    docs_x = []
    docs_y = []

    # for loop for filling all the above lists
    for intent in data["intents"]:
        for pattern in intent["pattern"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    # [0,0,0,1]
    # "hello", "no", "bye", "help"
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle", "wb") as f:
        #only lists we need for our model
        pickle.dump((words, labels, training, output), f)

# A neural network is created here. This class will create a certain amount of layers. Depending on how large you want your AI to be.
# I used 5 layers. The first layer is for the input layer. Than 3 layers of 16 neurons. And the last layer is for the output.
# Each neuron is connected with every neuron of the next layer. These will predict the possibilities of each 'tag'.
# For example: Input: Hello!. This word 'technically' goes through each layer, comparing to each pattern. greeting tag: 80.4%, goodbye tag:50.2%
# -> greeting tag will be chosen and a random answer will be used as response
# layer 1(input) layer 2         layer 3          layer 4          layer 5(output)
#   O       -       O       -       O        -       O        -       O
#   O       -       O       -       O        -       O        -       O
#   .               .               .                .                .
#   .               .               .                .                .
#   0       -       O       -       O        -       O        -       O 
#   0       -       O       -       O        -       O        -       O

class ChatModel(nn.Module):
    def __init__(self):
        super(ChatModel, self).__init__()  # Call the parent class constructor
        self.fc1 = nn.Linear(len(training[0]), 128)  
        self.fc2 = nn.Linear(128, 128) 
        self.fc3 = nn.Linear(128, 128)  
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, len(output[0]))


    # ?
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.softmax(self.fc5(x), dim=1)
        return x



# network is assigned and created
model = ChatModel()
# This calculates the cross entropy loss of the predicted outcome and the true target
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
except FileNotFoundError:
    # If model doesn't exist, train the model
    train_losses = []
    val_losses = []
    num_epochs = 5000
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(training, output, test_size=0.2, random_state=42)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass and compute training loss
        outputs = model(torch.Tensor(X_train))
        loss = criterion(outputs, torch.max(torch.Tensor(y_train), 1)[1])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Compute validation loss
        with torch.no_grad():
            val_outputs = model(torch.Tensor(X_val))
            val_loss = criterion(val_outputs, torch.max(torch.Tensor(y_val), 1)[1])
            val_losses.append(val_loss.item())

        # Save the model checkpoint
        torch.save(model.state_dict(), "model.pth")

    # Plot both training and validation losses
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# returns an array of all the words in each pattern/tag
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = (1)
    
    return np.array(bag)

tokenizer_gpt = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model_gpt = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat_history_ids = None

def chat(message):
    while True:
        # inp = input("you: ")
        # if inp.lower() == "quit":
        #     break
        print(message)
   
      
        input_tensor = torch.from_numpy(np.array([bag_of_words(message, words)]).astype(np.float32))
        output = model(input_tensor)

        # Find the index with the highest probability
        probabilities = torch.softmax(output, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        tag = labels[predicted_class]
        intent = next((intent for intent in data["intents"] if intent["tag"] == tag), None)
        print(max_prob)
        if intent and max_prob.item() > 0.1:
            responses = intent["responses"]
            response = random.choice(responses)  # Select a random response

        else:
        # When an input doesn't match any patterns then it will switch to this part. 
        # DialoGPT is a free pretrained model from microsoft that handles simple conversational tasks.
        # Trained on 147 Million reddit posts. Any inapropriate language is stopped by the model.
        # You can find explanation on HuggingFace.co
            counter = 0
            while counter >= 0:
                new_user_input_ids = tokenizer_gpt.encode(message + tokenizer_gpt.eos_token, return_tensors='pt')
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if counter > 0 else new_user_input_ids

                chat_history_ids = model_gpt.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer_gpt.eos_token_id)
                return tokenizer_gpt.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True) + " - " + str(max_prob)

        return response + " - " + str(max_prob)

