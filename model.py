import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
import copy
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, cnn):
        super(EncoderCNN, self).__init__()
        if cnn == 'resnet101':
            encoder = models.resnet101(pretrained=True)
        elif cnn == 'vgg19':
            encoder = models.vgg19(pretrained=True)
        for param in encoder.parameters():
            param.requires_grad_(False)
        
        modules = list(encoder.children())[:-1]        
        self.encoder = nn.Sequential(*modules)
        if cnn == 'resnet101':
            self.linear = nn.Linear(encoder.fc.in_features, embed_size)
        elif cnn == 'vgg19':
            self.linear = nn.Linear(encoder.classifier[6].in_features, embed_size)
        self.bn1 = nn.BatchNorm1d(embed_size)        
        
    def forward(self, images):
        features = self.encoder(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn1(features)
        
        return features


    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self, num_freezed):
        for l, child in enumerate(self.encoder.children()):
            if l > num_freezed:
                for param in child.parameters():
                  param.requires_grad = True

class DecoderRNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, vocab_size, num_layers=1, dropout=0, non_trainable=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers=num_layers
        self.word_embeddings, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, non_trainable)
        #self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)        
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout, 
                            batch_first=True, 
                            bidirectional=False)        
        
    def init_hidden(self, batch_size):        
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), \
                                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, features, captions):
        captions = captions[:, :-1]     
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embeddings(captions)       
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)                      
        lstm_out, self.hidden = self.lstm(inputs,self.hidden)
        outputs=self.linear(lstm_out)       
        return outputs 

    def greedy_sample(self, inputs):        
        cap_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size) 

        max_len = 0

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.linear(lstm_out)  
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            cap_output.append(max_idx.cpu().numpy()[0].item())             
            if (max_idx == 1):
                break
            
            inputs = self.word_embeddings(max_idx) 
            inputs = inputs.unsqueeze(1)

            max_len += 1
            if (max_len) == 20:
                break

        return cap_output    

    def beam(self, inputs):

        k = 10     
        cap_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size)
        
        # generating words next after CNN's vector
        lstm_out, hidden = self.lstm(inputs, hidden) 
        outputs = self.linear(lstm_out)  
        outputs = outputs.squeeze(1)
        outputs = F.log_softmax(outputs, dim=1)
        top_first_k = torch.topk(outputs, k, dim=1)

        # we will store scores, indexes (in vocab), their embeddings
        # and hiddens states in separate lists and we will use their orders 
        scores = top_first_k[0].squeeze(0)
        indexes = top_first_k[1].squeeze(0)
        embeddings = [self.word_embeddings(idx.unsqueeze(0)).unsqueeze(1) for idx in indexes]

        # hiddens are the same for k generated words but it will more
        # convinient to use them in the same 'style' as other objects 
        hiddens = [hidden]*k

        # collecting sentences
        sentences = [[index] for index in indexes]

        # startin length of each sentence is 1 now
        length = 1
        
        while True:

            length += 1

            current_scores = torch.tensor([])
            current_indexes = torch.tensor([], dtype=int)
            current_hiddens = []

            for i in range(k):
                
                # we get embedds and hiddens for each child
                h = hiddens[i]
                e = embeddings[i]

                # the same steps
                lstm_out, h_out = self.lstm(e, h)    
                outputs = self.linear(lstm_out)
                outputs = outputs.squeeze(1)
                outputs = F.log_softmax(outputs, dim=1)
                top_k = torch.topk(outputs, k, dim=1)

                temp_scores = top_k[0].squeeze(0)
                temp_indexes = top_k[1].squeeze(0)

                # for each child we add score of their parent score 
                temp_scores += scores[i]

                current_scores = torch.cat((current_scores, temp_scores))
                current_indexes = torch.cat((current_indexes, temp_indexes))
                current_hiddens.extend([h_out]*k)

            
            candidates = torch.topk(current_scores, k)[1] # indexes in arrays for best childs
            best_candidates_indexes = current_indexes[candidates] # indexes in vocab -||-
            best_candidates_scores = current_scores[candidates] # their scores
            best_hiddens = [current_hiddens[candidate] for candidate in candidates] 

            scores = best_candidates_scores # updating scores
            indexes = best_candidates_indexes # updating indexes (to generate next words)
            embeddings = [self.word_embeddings(idx.unsqueeze(0)).unsqueeze(1) for idx in indexes]
            hiddens = best_hiddens

            # extending current sentences by new words
            temp = []
            for i, idx in enumerate(candidates):
                sts = copy.deepcopy(sentences[idx//k])
                temp.append(sts)
                temp[i].append(current_indexes[idx])

            # updatinf current sentences
            sentences = temp      

            if length == 20:
                break
            
        # deviding score to length of the sentence
        normalized_score = []
        for i, score in enumerate(scores):
            try:
                score /= sentences[i].index(1)
            except:
                score /= len(sentences[i]) # if we don't get by generating <eos> token
            normalized_score.append(float(score))


        print('\nMEAN: ', np.mean(normalized_score))
        print('STD: ', np.std(normalized_score))
        print('Normalized score:', max(normalized_score))

        # choosing the best
        best_score = np.argmax(np.array(normalized_score))
        best_sentence = sentences[best_score]

        # returning truncated sentence 
        try:
            best_sentence = best_sentence[:best_sentence.index(1)]
        except:
            best_sentence = best_sentence
        best_sentence =  [int(word) for word in best_sentence]

        return best_sentence


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(weights_matrix), freeze=False)
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
