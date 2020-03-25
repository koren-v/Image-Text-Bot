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

from model import EncoderCNN, DecoderRNN, LanguageTransformer


# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


def gen_nopeek_mask(length):
    mask = torch.triu(torch.ones(length, length)).permute(1, 0)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__  == "__main__":

    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument("num_epochs", type=int,
                        help="num of epoches")
    parser.add_argument("--encoder_lr", type=float)
    parser.add_argument("--decoder_lr", type=float)
    parser.add_argument("--stage")
    parser.add_argument("--cnn")
    parser.add_argument("-val", "--make_validation", action="store_true")
    parser.add_argument("--vocab_from_file", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--unfreeze_encoder", action="store_true")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    args = parser.parse_args()


    embed_size=512
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 512
    if args.num_layers:
        num_layers = args.num_layers
    else:
        num_layers = 1

    vocab_threshold = 3
    vocab_from_file = args.vocab_from_file
    #hidden_size = 1024

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

    if args.cnn:
        cnn = args.cnn
    else:
        cnn = 'resnet101'
    encoder=EncoderCNN(embed_size, cnn)
    encoder=encoder.to(device)

    # decoder=DecoderRNN(embed_size=512, hidden_size=768 , vocab_size=vocab_size, num_layers=num_layers)
    # decoder=decoder.to(device)

    decoder = LanguageTransformer(vocab_size = vocab_size, embedding_size = 512, nhead = 8, num_encoder_layers = 6,
                                num_decoder_layers = 6, dim_feedforward = 2048, max_seq_length = 100,
                                pos_dropout = 0.1, trans_dropout = 0.1)

    decoder=decoder.to(device)


    if args.load_model:
        name = input('Type name of encoder/decoder')   
        if torch.cuda.is_available():
            encoder.load_state_dict(torch.load('./models/encoder'+name+'.pth'))
            decoder.load_state_dict(torch.load('./models/decoder'+name+'.pth'))
        else:
            encoder.load_state_dict(torch.load('./models/encoder'+name+'.pth', 
                                            map_location=torch.device('cpu')))
            decoder.load_state_dict(torch.load('./models/encoder'+name+'.pth', 
                                            map_location=torch.device('cpu')))

    num_epochs = args.num_epochs           
    save_every = 1
    save_every_step = 250            
    print_every = 100

    criterion = nn.CrossEntropyLoss()

    if args.unfreeze_encoder:
        encoder.unfreeze_encoder()
        encoder_params = encoder.parameters()
    else:
        encoder_params = list(encoder.linear.parameters()) + list(encoder.bn1.parameters())

    decoder_lr = args.decoder_lr
    encoder_lr = args.encoder_lr

    optimizer = torch.optim.Adam(
        [
            {"params":decoder.parameters(),"lr": decoder_lr},
            {"params":encoder_params, "lr": encoder_lr},
        
    ])

    #optimizer = torch.optim.Adam(params,lr=learning_rate)
    
    total_step = math.ceil(len(train_data_loader.dataset.caption_lengths) / train_data_loader.batch_sampler.batch_size)
    total_val_step = math.ceil(len(val_data_loader.dataset.caption_lengths) / val_data_loader.batch_sampler.batch_size)

    print('Start Training!')
    print(args.make_validation)


    if args.stage:
        stage = args.stage
    else:
        stage = 'default_stage'


    for epoch in range(1, num_epochs+1):

        train_running_loss = 0.0
        eval_running_loss = 0.0
        batches_skiped = 0

        for i_step in range(1, total_step+1):
            
            # choose batch_size indexes
            indices = train_data_loader.dataset.get_train_indices()        
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_data_loader.batch_sampler.sampler = new_sampler

            try:
                images, captions, tgt_key_padding_mask = next(iter(train_data_loader))
            except:
                batches_skiped+=1
                continue

            images = images.to(device)
            captions = captions.to(device)
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)

            decoder.zero_grad()
            encoder.zero_grad()

            features = encoder(images)
            features = features.to(device)

            captions_inp, captions_out = captions[:, :-1], captions[:, 1:].contiguous()

            tgt_mask = gen_nopeek_mask(captions_inp.shape[1])
            tgt_mask = tgt_mask.to(device)

            outputs = decoder(features, captions_inp, tgt_key_padding_mask[:, :-1], tgt_mask)
            outputs = outputs.to(device)

            loss = criterion(outputs.view(-1, vocab_size), captions_out.reshape(-1))
            loss.backward()
            optimizer.step()

            # metrics
            train_running_loss += loss.item() * outputs.size(0) #I added this part =)
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, num_epochs, i_step, total_step, loss.item())
            
            print('\r' + stats, end="")
            sys.stdout.flush()
            if i_step % print_every == 0:
                print('\r' + stats)



        if args.make_validation:

            print('\nStart validation!')

            with torch.no_grad():

                for i_step in range(1, total_val_step+1): 
        
                    indices = val_data_loader.dataset.get_train_indices()        
                    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
                    val_data_loader.batch_sampler.sampler = new_sampler

                    try:
                        images, captions, tgt_key_padding_mask = next(iter(train_data_loader))
                    except:
                        continue

                    images = images.to(device)
                    captions = captions.to(device)
<<<<<<< HEAD
                    tgt_key_padding_mask = tgt_key_padding_mask.to(device)
=======
>>>>>>> parent of e7c3925... mask to device

                    features = encoder(images)
                    features = features.to(device)

                    captions_inp, captions_out = captions[:, :-1], captions[:, 1:]

                    tgt_mask = gen_nopeek_mask(captions_inp.shape[1])
                    tgt_mask = tgt_mask.to(device)

                    outputs = decoder(features, captions_inp, tgt_key_padding_mask[:, :-1], tgt_mask)
                    outputs = outputs.to(device)

                    loss = criterion(outputs.view(-1, vocab_size), captions_out.reshape(-1))
                    eval_running_loss += loss.item() * outputs.size(0) #I added this part =)
                    stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch, num_epochs, i_step, total_val_step, loss.item())
                    print('\r' + stats, end="")
                    sys.stdout.flush()
                    if i_step % print_every == 0:
                        print('\r' + stats)

        train_epoch_loss = train_running_loss / len(train_data_loader.dataset.caption_lengths)
        eval_epoch_loss = eval_running_loss / len(val_data_loader.dataset.caption_lengths)

        if epoch % save_every == 0:
            torch.save(decoder.state_dict(),
                        os.path.join('./models', 'decoder_%d_%.2f_v_%.2f_t_%s.pth' % (epoch, 
                                                                                        eval_epoch_loss,
                                                                                        train_epoch_loss, 
                                                                                        stage)))
            torch.save(encoder.state_dict(),
                        os.path.join('./models', 'encoder_%d_%.2f_v_%.2f_t_%s.pth' % (epoch, 
                                                                                        eval_epoch_loss, 
                                                                                        train_epoch_loss, 
                                                                                        stage)))

        print('\nTrain loss: ', train_epoch_loss)
        print('Eval loss: ', eval_epoch_loss)
        print('Batches skipped during training: ', batches_skiped)