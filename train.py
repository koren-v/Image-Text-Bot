import os
import sys
import argparse

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__  == "__main__":

    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument("num_epochs", type=int,
                        help="num of epoches")
    parser.add_argument("lr", type=float)
    parser.add_argument("-val", "--make_validation", action="store_true")
    parser.add_argument("--vocab_file", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    parser.add_argument("--sgd", action="store_true")
    args = parser.parse_args()


    embed_size=512
    batch_size = 512    
    vocab_threshold = 3
    vocab_from_file = args.vocab_file
    hidden_size = 1024

    train_data_loader = get_loader(transform=transform_train,
                                mode='train',
                                batch_size=batch_size,
                                vocab_threshold=vocab_threshold,
                                vocab_from_file=vocab_from_file)

    val_data_loader = get_loader(transform=transform_train,
                                mode='val',
                                batch_size=batch_size,
                                vocab_from_file=True)                            

    vocab_size = len(train_data_loader.dataset.vocab)

    encoder=EncoderCNN(embed_size)
    encoder=encoder.to(device)

    decoder=DecoderRNN(embed_size=512, hidden_size=1024 , vocab_size=vocab_size, num_layers=1)
    decoder=decoder.to(device)


    if args.load_model:
        name = input('Type name of encoder/decoder')
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load('./models/encoder'+name+'.pkl'))
            decoder.load_state_dict(torch.load('./models/decoder'+name+'.pkl'))
        else:
            encoder.load_state_dict(torch.load('./models/encoder'+name+'.pkl', 
                                            map_location=torch.device('cpu')))
            decoder.load_state_dict(torch.load('./models/encoder'+name+'.pkl', 
                                            map_location=torch.device('cpu')))

    num_epochs = args.num_epochs           
    save_every = 1
    save_every_step = 250            
    print_every = 100

    criterion = nn.CrossEntropyLoss()

    if args.unfreeze_encoder:
        encoder.unfreeze_encoder()
        params = list(decoder.parameters()) + list(encoder.parameters())
    else:
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn1.parameters())

    learning_rate=args.lr
    if args.sgd:
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(params,lr=learning_rate)
    
    total_step = math.ceil(len(train_data_loader.dataset.caption_lengths) / train_data_loader.batch_sampler.batch_size)
    total_val_step = math.ceil(len(val_data_loader.dataset.caption_lengths) / val_data_loader.batch_sampler.batch_size)

    print('Start Training!')
    print(args.make_validation)

    for epoch in range(1, num_epochs+1):

        train_running_loss = 0.0
        eval_running_loss = 0.0
        batches_skiped = 0

        for i_step in range(1, total_step+1):

            indices = train_data_loader.dataset.get_train_indices()        
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_data_loader.batch_sampler.sampler = new_sampler

            try:
                images, captions = next(iter(train_data_loader))
            except:
                batches_skiped+=1
                continue

            images = images.to(device)
            captions = captions.to(device)

            decoder.zero_grad()
            encoder.zero_grad()

            features = encoder(images)
            features = features.to(device)

            outputs = decoder(features, captions)
            outputs = outputs.to(device)

            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()

            # metrics
            train_running_loss += loss.item() * outputs.size(0) #I added this part =)
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, num_epochs, i_step, total_step, loss.item())
            
            print('\r' + stats, end="")
            sys.stdout.flush()
            if i_step % print_every == 0:
                print('\r' + stats)
            
            # saving models
            if i_step % save_every_step == 0:
                torch.save(decoder.state_dict(), os.path.join('./models', 'decoder_%d_stp_%.2f_loss.pth' % (i_step,loss.item())))
                torch.save(encoder.state_dict(), os.path.join('./models', 'encoder_%d_stp_%.2f_loss.pth' % (i_step,loss.item())))

        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder_%d.pth' % epoch))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder_%d.pth' % epoch))


        if args.make_validation:

            print('Start validation!')

            with torch.no_grad():

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

                    features = encoder(images)
                    features = features.to(device)

                    outputs = decoder(features, captions)
                    outputs=outputs.to(device)
                    
                    loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                    eval_running_loss += loss.item() * outputs.size(0) #I added this part =)
                    stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, num_epochs, i_step, total_val_step, loss.item())
                    print('\r' + stats, end="")
                    sys.stdout.flush()
                    if i_step % print_every == 0:
                        print('\r' + stats)


        train_epoch_loss = train_running_loss / len(train_data_loader.dataset.caption_lengths)
        eval_epoch_loss = eval_running_loss / len(val_data_loader.dataset.caption_lengths)
        print('\nTrain loss: ', train_epoch_loss)
        print('Eval loss: ', eval_epoch_loss)
        print('Batches skipped during training: ', batches_skiped)
