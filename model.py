import torch
import torch.nn as nn
import torchvision.models as models
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]        
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)        
        self.bn1 = nn.BatchNorm1d(embed_size)        
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn1(features)
        
        return features

    def freeze_encoder(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.resnet.parameters():
            param.requires_grad = True

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers=num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)        
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True, 
                            bidirectional=False)        
        
    def init_hidden(self, batch_size):        
        return torch.zeros(1, batch_size, self.hidden_size).to(device), \
                torch.zeros(1, batch_size, self.hidden_size).to(device)

    def forward(self, features, captions):
        captions = captions[:, :-1]  #== we drop last token in input example
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        embeds = self.word_embeddings(captions)       
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)   #after getting vectors for each token we                     
        lstm_out, self.hidden = self.lstm(inputs,self.hidden)           #add to begining of input vector from CNN
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
            _, max_idx = torch.max(outputs, dim=1) #argmax with no _
            cap_output.append(max_idx.cpu().numpy()[0].item())             
            if (max_idx == 1):
                break
            
            inputs = self.word_embeddings(max_idx) 
            inputs = inputs.unsqueeze(1)

            max_len += 1
            if (max_len) == 20:
                break

        return cap_output    

#     def beam_decode(self, decoder_hiddens, encoder_outputs):

#         beam_width = 10
#         topk = 1
#         decoded_batch = []

#         for idx in range(999):
#             hidden = self.init_hidden(batch_size)


# class BeamSearchNode(object):
#     def __init__(self, hiddenstate, previusNode, wordId, logProb, length):

#         self.h = hiddenstate,
#         self.previusNode = previusNode
#         self.wordId = wordId
#         self.logp = logProb
#         self.leng = length

#     def eval(self):
#         return self.logp / float(self.leng -1 + 1e-6)





# class DecoderTransform(nn.Module):
#     def __init__(self, embed_size, vocab_size, num_layers=1):
#         super(DecoderTransform, self).__init__()
#         #self.hidden_size = hidden_size
#         self.vocab_size = vocab_size
#         self.num_layers=num_layers
#         self.word_embeddings = nn.Embedding(vocab_size, embed_size)
#         #self.linear = nn.Linear(hidden_size, vocab_size)        
#         self.transformer = nn.Transformer()

#     def forward(self, features, captions):
#         import pdb
#         pdb.set_trace()
#         target_caption = copy.deepcopy(captions)
#         captions = captions[:, :-1]    
#         self.batch_size = features.shape[0]
#         self.hidden = self.init_hidden(self.batch_size)
#         embeds = self.word_embeddings(captions)
#         target = self.word_embeddings(target_caption)     
#         inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)                      
#         output = self.transformer(inputs, target)
#         #outputs=self.linear(output)       
#         return output 