import torch
from torchvision import transforms
from data_loader import get_loader
import matplotlib.pyplot as plt
import numpy as np

from model import EncoderCNN, DecoderRNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clean_sentence(output):
    sentence = ''
    for x in output[1:]:
        if x == 1:
            break
        sentence += ' ' + data_loader.dataset.vocab.idx2word[x]
    return sentence

def get_prediction():
       
    orig_image, image = next(iter(data_loader))
    #plt.imshow(np.squeeze(orig_image))
    #plt.title('Sample Image')
    #plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    print('Shape of feat: ', features.shape)
    output = decoder.greedy_sample(features)    
    sentence = clean_sentence(output)
    print(sentence)



if __name__ == "__main__":

    transform_test = transforms.Compose([ 
                        transforms.Resize(256),                          
                        transforms.RandomCrop(224),                     
                        transforms.RandomHorizontalFlip(),              
                        transforms.ToTensor(),                          
                        transforms.Normalize((0.485, 0.456, 0.406),     
                                            (0.229, 0.224, 0.225))])

    data_loader = get_loader(transform=transform_test,    
                                mode='test',
                                cocoapi_loc='')


    embed_size=512   
    vocab_threshold = 5
    hidden_size = 1024


    encoder=EncoderCNN(embed_size)
    decoder=DecoderRNN(embed_size=512, hidden_size=1024 , vocab_size=8856, num_layers=1)

    if torch.cuda.is_available():
        encoder.load_state_dict(torch.load('./models/encoder-1.pkl'))
        decoder.load_state_dict(torch.load('./models/decoder-1.pkl'))
    else:
        encoder.load_state_dict(torch.load('./models/encoder-1.pkl', 
                                        map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load('./models/decoder-1.pkl', 
                                        map_location=torch.device('cpu')))


    encoder=encoder.to(device)
    decoder=decoder.to(device)

    encoder.eval()
    decoder.eval()

    print("Start making prediction ...")

    get_prediction()