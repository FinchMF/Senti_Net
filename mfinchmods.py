import time
import sys
import numpy as np
from collections import Counter



#######################
# N E U R A L N E T 1 #
#######################


class Senti_Net():
    def __init__(self, reviews,labels,min_count = 10,polarity_cutoff = 0.1,hidden_nodes = 10, learning_rate = 0.1):
        # Assign a seed for reproducibility
        np.random.seed(1)

        # process text data
        self.pre_process_data(reviews, labels, polarity_cutoff, min_count)
        
        # Build network 
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    # preprocessing function
    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):
        
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if labels[i] == 'POSITIVE':
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term,cnt in list(total_counts.most_common()):
            if(cnt >= 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in pos_neg_ratios.most_common():
            if(ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))


        # populate review_vocab with all words
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                # create polarity checks - this is to ignore words that are not used as often, 
                # or too often (like articles and periods) thus generate noise for the machine
                if total_counts[word] > min_count:
                    if word in pos_neg_ratios.keys():
                        if (pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)

        # convert the vocab set to list in order to call words via indices
        self.review_vocab = list(review_vocab)
        
        # populate label_vocab with all of the words in given label
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # convert the label vocab set to list in order to call labels via indices
        self.label_vocab = list(label_vocab)
        
        # store the sizes of the review and label vocabs.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # create a dictionary of words in the vocab mapped to index positions
        self.word2dict = {}
        for i, word in enumerate(self.review_vocab):
            self.word2dict[word] = i
        
        # create a dictionary of labels mapped to index positions
        self.label2dict = {}
        for i, label in enumerate(self.label_vocab):
            self.label2dict[label] = i
    # initialize neural network
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        # initialize weights
        # weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

        # weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
       
        # The input layer, a two-dimensional matrix with shape 1 x hidden_nodes
        self.layer_1 = np.zeros((1,hidden_nodes))
    
    
    def convert_label_to_target(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
    # build sigmoid activation funtion (other activation functions may also be available) 
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    # convert the sigmoid out to a derivative
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
   
    def train(self, training_reviews_raw, training_labels):

        #pre-process training reviews in order to deal directly with the indices of non-zero inputs
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if word in self.word2dict.keys():
                    indices.add(self.word2dict[word])
            training_reviews.append(list(indices))

        # confirm our matrices are the same size
        assert(len(training_reviews) == len(training_labels))
        # keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        # track start for printing time statistics
        start = time.time()
        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            # get the next review and respective label
            review = training_reviews[i]
            label = training_labels[i]
            
            ### Forward pass ###
            # Hidden layer
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))            

            ### Backward pass ###
            # Output error
            layer_2_error = layer_2 - self.convert_label_to_target(label) 
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity - remains the same 
            # update the weights
            # update output-to-hidden with gradient
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate
            # update input-to-hidden with gradient
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate 

            # track correct predictions
            if layer_2 >= 0.5 and label == 'POSITIVE':
                correct_so_far += 1
            elif layer_2 < 0.5 and label == 'NEGATIVE':
                correct_so_far += 1
            
            # print out our prediction accuracy and speed 
            # throughout the training process. 
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rHow much I've read:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% How fast I can read:(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def predict(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # track how many correct predictions
        correct = 0

        # time number of predictions per second 
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        predictions = []
        for i in range(len(testing_reviews)):
            pred = self.classify(testing_reviews[i])
            predictions.append(pred)
            if pred == testing_labels[i]:
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rHow much I've read:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% How fast I can read:(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
        return dict(zip(predictions, testing_labels))

    def classify(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network
        
        # Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2dict.keys():
                unique_indices.add(self.word2dict[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
         
        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # return NEGATIVE for other values
        if layer_2[0] >= 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"







#######################
# N E U R A L N E T 2 #
#######################





class Tw_Net():
    def __init__(self, tweets,labels,min_count = 10,polarity_cutoff = 0.1,hidden_nodes = 10, learning_rate = 0.1):
        # Assign a seed for reproducibility
        np.random.seed(1)

        # process text data
        self.pre_process_data(tweets, labels, polarity_cutoff, min_count)
        
        # Build network 
        self.init_network(len(self.tweet_vocab),hidden_nodes, 1, learning_rate)

    # preprocessing function
    def pre_process_data(self, tweets, labels, polarity_cutoff, min_count):
        
        dem_counts = Counter()
        rep_counts = Counter()
        total_counts = Counter()

        for i in range(len(tweets)):
            if labels[i] == labels[0]:
                for word in tweets[i].split(" "):
                    dem_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in tweets[i].split(" "):
                    rep_counts[word] += 1
                    total_counts[word] += 1

        dem_rep_ratios = Counter()

        for term,cnt in list(total_counts.most_common()):
            if(cnt >= 50):
                dem_rep_ratio = dem_counts[term] / float(rep_counts[term]+1)
                dem_rep_ratios[term] = dem_rep_ratio

        for word,ratio in dem_rep_ratios.most_common():
            if(ratio > 1):
                dem_rep_ratios[word] = np.log(ratio)
            else:
                dem_rep_ratios[word] = -np.log((1 / (ratio + 0.01)))


        # populate review_vocab with all words
        tweet_vocab = set()
        for tweet in tweets:
            for word in tweet.split(" "):
                # create polarity checks - this is to ignore words that are not used as often, 
                # or too often (like articles and periods) thus generate noise for the machine
                if total_counts[word] > min_count:
                    if word in dem_rep_ratios.keys():
                        if (dem_rep_ratios[word] >= polarity_cutoff) or (dem_rep_ratios[word] <= -polarity_cutoff):
                            tweet_vocab.add(word)
                    else:
                        tweet_vocab.add(word)

        # convert the vocab set to list in order to call words via indices
        self.tweet_vocab = list(tweet_vocab)
        
        # populate label_vocab with all of the words in given label
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # convert the label vocab set to list in order to call labels via indices
        self.label_vocab = list(label_vocab)
        
        # store the sizes of the review and label vocabs.
        self.tweet_vocab_size = len(self.tweet_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # create a dictionary of words in the vocab mapped to index positions
        self.word2dict = {}
        for i, word in enumerate(self.tweet_vocab):
            self.word2dict[word] = i
        
        # create a dictionary of labels mapped to index positions
        self.label2dict = {}
        for i, label in enumerate(self.label_vocab):
            self.label2dict[label] = i
    # initialize neural network
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        # initialize weights
        # weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))

        # weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
       
        # The input layer, a two-dimensional matrix with shape 1 x hidden_nodes
        self.layer_1 = np.zeros((1,hidden_nodes))
    
    
    def convert_label_to_target(self,label):
        if(label == 'democrat'):
            return 1
        else:
            return 0
    # build sigmoid activation funtion (other activation functions may also be available) 
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    # convert the sigmoid out to a derivative
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
    
   
    def train(self, training_tweets_raw, training_labels):

        #pre-process training reviews in order to deal directly with the indices of non-zero inputs
        training_tweets = list()
        for tweet in training_tweets_raw:
            indices = set()
            for word in tweet.split(" "):
                if word in self.word2dict.keys():
                    indices.add(self.word2dict[word])
            training_tweets.append(list(indices))

        # confirm our matrices are the same size
        assert(len(training_tweets) == len(training_labels))
        # keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        # track start for printing time statistics
        start = time.time()
        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_tweets)):
            # get the next review and respective label
            tweet = training_tweets[i]
            label = training_labels[i]
            
            ### Forward pass ###
            # Hidden layer
            self.layer_1 *= 0
            for index in tweet:
                self.layer_1 += self.weights_0_1[index]
            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))            

            ### Backward pass ###
            # Output error
            layer_2_error = layer_2 - self.convert_label_to_target(label) 
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity - remains the same 
            # update the weights
            # update output-to-hidden with gradient
            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate
            # update input-to-hidden with gradient
            for index in tweet:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate 

            # track correct predictions
            if layer_2 >= 0.5 and label == 'democrat':
                correct_so_far += 1
            elif layer_2 < 0.5 and label == 'republican':
                correct_so_far += 1
            
            # print out our prediction accuracy and speed 
            # throughout the training process. 
            elapsed_time = float(time.time() - start)
            tweets_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rHow much I've read:" + str(100 * i/float(len(training_tweets)))[:4] \
                             + "% How fast I can read:(reviews/sec):" + str(tweets_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def predict(self, testing_tweets, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # track how many correct predictions
        correct = 0

        # time number of predictions per second 
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        predictions = []
        for i in range(len(testing_tweets)):
            pred = self.classify(testing_tweets[i])
            predictions.append(pred)
            if pred == testing_labels[i]:
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            tweets_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rHow much I've read:" + str(100 * i/float(len(testing_tweets)))[:4] \
                             + "% How fast I can read:(reviews/sec):" + str(tweets_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
        return dict(zip(predictions, testing_labels))
    
    def classify(self, tweet):
        
        # Run a forward pass through the network
        
        # Hidden layer
        self.layer_1 *= 0
        unique_indices = set()
        for word in tweet.lower().split(" "):
            if word in self.word2dict.keys():
                unique_indices.add(self.word2dict[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]
        
        # Output layer
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))
         
        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # return NEGATIVE for other values
        if layer_2[0] >= 0.5:
            return "a democrat tweeted this"
        else:
            return "a republican tweeted this"
