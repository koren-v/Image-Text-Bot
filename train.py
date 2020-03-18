import os
import sys
sys.path.append('../Img2Txt')

from pycocotools.coco import COCO

import nltk

from data_loader import get_loader
from torchvision import transforms

import math
import numpy as np

import torch.utils.data as data

import torch
import torch.nn as nn
import torchvision.models as models

from model import EncoderCNN, DecoderRNN


# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


# Set the minimum word count threshold.
vocab_threshold = 5

# Specify the batch size.
#batch_size = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__  == "__main__":

    nltk.download('punkt')


    embed_size=512
    batch_size = 512    
    vocab_threshold = 5
    vocab_from_file = True
    hidden_size = 1024

    train_data_loader = get_loader(transform=transform_train,
                                mode='train',
                                batch_size=batch_size,
                                vocab_threshold=vocab_threshold,
                                vocab_from_file=vocab_from_file,
                                cocoapi_loc='')

    val_data_loader = get_loader(transform=transform_train,
                                mode='train',
                                batch_size=batch_size,
                                vocab_threshold=vocab_threshold,
                                vocab_from_file=vocab_from_file,
                                cocoapi_loc='')                            


    # indices = data_loader.dataset.get_train_indices()
    vocab_size = len(train_data_loader.dataset.vocab)

    # new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    # data_loader.batch_sampler.sampler = new_sampler


    encoder=EncoderCNN(embed_size)
    encoder=encoder.to(device)

    # decoder=DecoderRNN(embed_size=512, hidden_size=1024 , vocab_size=8856, num_layers=1)
    decoder=DecoderRNN(embed_size=512, hidden_size=1024 , vocab_size=vocab_size, num_layers=1)
    decoder=decoder.to(device)


    num_epochs = 15           
    save_every = 1
    save_every_step = 250            
    print_every = 100         
    log_file = 'training_log.txt'

    make_validation = False


    criterion = nn.CrossEntropyLoss()

    #we don't train resnet
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn1.parameters())

    learning_rate=0.006
    optimizer = torch.optim.Adam(params,lr=learning_rate)
    
    total_step = math.ceil(len(train_data_loader.dataset.caption_lengths) / train_data_loader.batch_sampler.batch_size)
    total_val_step = math.ceil(len(val_data_loader.dataset.caption_lengths) / val_data_loader.batch_sampler.batch_size)


    #f = open(log_file, 'w')
    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        for i_step in range(1, total_step+1):        
            indices = train_data_loader.dataset.get_train_indices()        
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_data_loader.batch_sampler.sampler = new_sampler

            try:
                images, captions = next(iter(train_data_loader))
            except:
                continue

            images = images.to(device)
            captions = captions.to(device)
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            features = features.to(device)
            outputs = decoder(features, captions)
            outputs=outputs.to(device)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            #loss = criterion(outputs.view(-1, 8856), captions.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * outputs.size(0) #I added this part =)
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
            print('\r' + stats, end="")
            sys.stdout.flush()
            # f.write(stats + '\n')
            # f.flush()
            if i_step % print_every == 0:
                print('\r' + stats)
            if i_step % save_every_step == 0:
                torch.save(decoder.state_dict(), os.path.join('./models', 'decoder_%d_stp.pth' % epoch))
                torch.save(encoder.state_dict(), os.path.join('./models', 'encoder_%d_stp.pth' % epoch))
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder_%d.pth' % epoch))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder_%d.pth' % epoch))

        if make_validation:
            for i_step in range(1, total_val_step+1):        
                indices = val_data_loader.dataset.get_train_indices()        
                new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
                val_data_loader.batch_sampler.sampler = new_sampler

                try:
                    images, captions = next(iter(val_data_loader))
                except:
                    continue

                images = images.to(device)
                captions = captions.to(device)
                decoder.zero_grad()
                encoder.zero_grad()
                features = encoder(images)
                features = features.to(device)
                outputs = decoder(features, captions)
                outputs=outputs.to(device)
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                #loss = criterion(outputs.view(-1, 8856), captions.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * outputs.size(0) #I added this part =)
                stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
                print('\r' + stats, end="")
                sys.stdout.flush()
                # f.write(stats + '\n')
                # f.flush()
                if i_step % print_every == 0:
                    print('\r' + stats)


        epoch_loss = running_loss
        print('My loss: ', epoch_loss)

    #f.close()