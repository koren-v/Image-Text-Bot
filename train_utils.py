import os
import sys
import math
import copy
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

chencherry = SmoothingFunction()


def collect_preds(raw_preds, outputs):
    if len(raw_preds) == 0:
      raw_preds = outputs.cpu().detach().numpy()
    else:
      raw_preds = np.vstack((raw_preds, outputs.cpu().detach().numpy()))
    return raw_preds
        
        
def compute_metric(captions, preds):
    preds = torch.argmax(preds, axis=2)
    assert preds.shape == captions.shape
    bleu_score = 0.0
    for pred, ground_truth in zip(preds, captions):
        bleu_score += sentence_bleu([ground_truth.tolist()], pred.tolist(), smoothing_function=chencherry.method1)
    return bleu_score/len(captions)


def gen_nopeek_mask(length):
    mask = torch.triu(torch.ones(length, length)).permute(1, 0)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def add_examples(captions_out, outputs, data_loader, num_examples=4):
    examples = ''
    data_loader = data_loader
    for ground_truth, prediction in zip(captions_out[:num_examples,:], outputs[:num_examples,:,:]):
        prediction = torch.argmax(prediction, axis=1)
        truth_text = [data_loader.dataset.vocab.idx2word[int(idx)] for idx in ground_truth]
        preds_text = [data_loader.dataset.vocab.idx2word[int(idx)] for idx in prediction]
        examples += ' '.join(truth_text) + '/' + ' '.join(preds_text) + '\n'
    return examples


def epoch(model, phase, device, criterion, optimizer, 
          data_loader, tb, grad_accumulation_step, scheduler=None):

    encoder = model['encoder']
    decoder = model['decoder']

    if phase == 'train':
        encoder.train()
        decoder.train()
    else: 
        encoder.eval()
        decoder.eval()

    running_loss = 0.0
    running_bleu = 0.0
    batches_skiped = 0
    
    vocab_size = len(data_loader.dataset.vocab)
    total_steps = math.ceil(len(data_loader.dataset.caption_lengths)\
                             / data_loader.batch_sampler.batch_size)

    for step in range(total_steps):
        
        indices = data_loader.dataset.get_train_indices()        
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        try:
            images, captions, tgt_key_padding_mask = next(iter(data_loader))
        except BaseException:
            batches_skiped += 1
            continue

        images = images.to(device)
        captions = captions.to(device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(device)

        with torch.set_grad_enabled(phase == 'train'):

            features = encoder(images)
            features = features.to(device)
            captions_inp, captions_out = captions[:, :-1], captions[:, 1:].contiguous()

            tgt_mask = gen_nopeek_mask(captions_inp.shape[1])
            tgt_mask = tgt_mask.to(device)

            outputs = decoder(features, captions_inp, tgt_key_padding_mask[:,1:], tgt_mask)
            outputs = outputs.to(device)

            loss = criterion(outputs.view(-1, vocab_size), captions_out.reshape(-1))

            examples = add_examples(captions_out, outputs, data_loader)
            tb.add_text('ground_truth/predictions', examples, step)

            if phase == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                if (step+1) % grad_accumulation_step == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

        running_loss += loss.item() * features.size(0)
        bleu4 = compute_metric(captions_out, outputs)
        running_bleu += bleu4

        stats = 'Step [{}/{}], Loss: {:.4f}, BLEU-4: {:.4f}'.format(step, total_steps, loss.item(), bleu4)

        print('\r' + stats, end="")
        sys.stdout.flush()

        tb.add_scalar('{} Batch Loss'.format(phase.title()), loss.item(), step)
        tb.add_scalar('{} Batch BLEU'.format(phase.title()), bleu4, step)
    
    epoch_loss = running_loss / len(data_loader.dataset.caption_lengths)  # len of the data
    epoch_bleu = running_bleu / total_steps
    epoch_dict = {'batches_skiped': batches_skiped,
                  'epoch_loss': epoch_loss, 
                  'epoch_bleu': epoch_bleu,
                  'encoder': encoder,
                  'decoder': decoder}
    return epoch_dict


def fit(model, criterion, optimizer, dataloader_dict,
        num_epochs, device, stage, hyper_params, scheduler=None,
        last_epoch=None, grad_accumulation_step=1):

    tb = SummaryWriter(comment=stage)
    train_loss = []
    valid_loss = []
    train_bleu = []
    valid_bleu = []

    for i in range(num_epochs):
        print('Epoch {}/{}'.format(i+1, num_epochs))
        print('*'*48)
        for phase in ['train', 'val']:

            epoch_dict = epoch(model, phase, device, criterion, optimizer, 
                               dataloader_dict[phase], tb, 
                               grad_accumulation_step = grad_accumulation_step, 
                               scheduler=scheduler)
            
            print('\n{} epoch loss: {:.4f} '.format(phase , epoch_dict['epoch_loss']))
            print('{} epoch metric: {:.4f} '.format(phase , epoch_dict['epoch_bleu']))

            tb.add_scalar('{} Epoch Loss'.format(phase.title()), epoch_dict['epoch_loss'], i)
            tb.add_scalar('{} Epoch BLEU'.format(phase.title()), epoch_dict['epoch_bleu'], i)

            # probably it is unnecessary lists
            if phase == 'train': 
                train_loss.append(epoch_dict['epoch_loss'])
                train_bleu.append(epoch_dict['epoch_bleu'])
            else: 
                valid_loss.append(epoch_dict['epoch_loss'])
                valid_bleu.append(epoch_dict['epoch_bleu'])

                model_name = '_{}_{:.2f}_val_{:.2f}_tr_{}.pth'.format(i+last_epoch if last_epoch else i,
                                                                      valid_loss[-1],
                                                                      train_loss[-1],
                                                                      stage)
                torch.save(epoch_dict['decoder'].state_dict(),
                                os.path.join('./models', 'decoder'+model_name))
                torch.save(epoch_dict['encoder'].state_dict(),
                                os.path.join('./models', 'encoder'+model_name))

        print()
    print('Batches skipped', epoch_dict['batches_skiped'])
    metrics = {'Train_Loss': train_loss[-1], 'Val_Loss': valid_loss[-1],
               'Train_Bleu': train_bleu[-1], 'Valid_Bleu': valid_bleu[-1]}

    tb.add_hparams(hyper_params, metrics)
    tb.close()
