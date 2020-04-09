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

class Learner():

    def __init__(self, model, criterion, optimizer, dataloader_dict,
        num_epochs, device, stage, hyper_params, scheduler=None, last_epoch=None, grad_acumulation_step = 1):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader_dict = dataloader_dict
        self.num_epochs = num_epochs
        self.device = device
        self.stage = stage
        self.hyper_params = hyper_params
        self.scheduler = scheduler
        self.last_epoch = last_epoch
        self.grad_acumulation_step = grad_acumulation_step
        self.tb = SummaryWriter(comment=self.stage)

    def fit(self):
        # tb.add_graph(model['encoder'])
        # tb.add_graph(model['decoder'])
        best_loss = float('inf')
        train_loss = []
        valid_loss = []
        train_bleu = []
        valid_bleu = []

        for i in range(self.num_epochs):
            print('Epoch {}/{}'.format(i+1, self.num_epochs))
            print('*'*48)
            for phase in ['train', 'val']:

                epoch_dict = self.epoch(phase)
                
                print('\n{} epoch loss: {:.4f} '.format(phase , epoch_dict['epoch_loss']))
                print('{} epoch metric: {:.4f} '.format(phase , epoch_dict['epoch_bleu']))

                self.tb.add_scalar('{} Epoch Loss'.format(phase.title()), epoch_dict['epoch_loss'], i)
                self.tb.add_scalar('{} Epoch BLEU'.format(phase.title()), epoch_dict['epoch_bleu'], i)

                # probably it is unnessesury lists
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
                    
                # not implemented loading best model weights callback
                # if phase == 'val' and best_loss>epoch_dict['epoch_loss']:
                #     best_encoder_wts = copy.deepcopy(epoch_dict['encoder'].state_dict())
                #     best_decoder_wts = copy.deepcopy(epoch_dict['decoder'].state_dict())
                #     best_loss = epoch_dict['epoch_loss']
            print()
        print('Batches skipped', epoch_dict['batches_skiped'])
        metrics = {'Train_Loss': train_loss[-1], 'Val_Loss': valid_loss[-1],
                'Train_Bleu': train_bleu[-1], 'Valid_Bleu': valid_bleu[-1]}

        self.tb.add_hparams(hyper_params, metrics)
        self.tb.close()

    def epoch(self, phase):

        encoder = self.model['encoder']
        decoder = self.model['decoder']

        if phase=='train': 
            encoder.train()
            decoder.train()
        else: 
            encoder.eval()
            decoder.eval()

        running_loss = 0.0
        running_bleu = 0.0
        batches_skiped = 0
        captionss = np.array([])
        raw_preds = np.array([])
        
        vocab_size = len(self.dataloader_dict[phase].dataset.vocab)
        total_steps = math.ceil(len(self.dataloader_dict[phase].dataset.caption_lengths)\
                                / self.dataloader_dict[phase].batch_sampler.batch_size)

        #optimizer.zero_grad()
        for step in range(total_steps):

            indices = self.dataloader_dict[phase].dataset.get_train_indices()        
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            self.dataloader_dict[phase].batch_sampler.sampler = new_sampler

            try:
                images, captions = next(iter(self.dataloader_dict[phase]))
            except:
                batches_skiped+=1
                continue

            images = images.to(self.device)
            captions = captions.to(self.device)

            with torch.set_grad_enabled(phase == 'train'):

                features = encoder(images)
                features = features.to(self.device)
                outputs = decoder(features, captions)

                loss = self.criterion(outputs.view(-1, vocab_size), captions.view(-1))

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    if (step+1)%self.grad_acumulation_step == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    # writting weights and grads to tensorboard's histogram    
                    for name, weight in decoder.named_parameters():
                        self.tb.add_histogram(name, weight, step)
                        self.tb.add_histogram(f'{name}.grad', weight.grad, step)
                elif step%(total_steps//5)==0:
                    examples = self.add_examples(captions, outputs, phase)
                    self.tb.add_text(f'{phase}:ground_truth/predictions', examples, step)

            running_loss += loss.item() * features.size(0)
            bleu4 = self.compute_metric(captions, outputs)
            running_bleu+=bleu4

            stats = 'Step [{}/{}], Loss: {:.4f}, BLEU-4: {:.4f}'.format(step, total_steps, loss.item(), bleu4)

            print('\r' + stats, end="")
            sys.stdout.flush()

            self.tb.add_scalar('{} Batch Loss'.format(phase.title()), loss.item(), step)
            self.tb.add_scalar('{} Batch BLEU'.format(phase.title()), bleu4, step)
        
        epoch_loss = running_loss / len(self.dataloader_dict[phase].dataset.caption_lengths) #len of the data
        epoch_bleu = running_bleu / total_steps

        if self.scheduler and phase=='val': self.scheduler.step(epoch_loss)

        epoch_dict = {'batches_skiped':batches_skiped,
                    'epoch_loss': epoch_loss, 
                    'epoch_bleu': epoch_bleu,
                    'encoder': encoder,
                    'decoder': decoder}
        return epoch_dict

    def add_examples(self, captions_out, outputs, phase, num_examples=4):
        examples = ''
        for ground_truth, prediction in zip(captions_out[:num_examples,:], outputs[:num_examples,:,:]):
            prediction = torch.argmax(prediction, axis=1)
            truth_text = [self.dataloader_dict[phase].dataset.vocab.idx2word[int(idx)] for idx in ground_truth]
            preds_text = [self.dataloader_dict[phase].dataset.vocab.idx2word[int(idx)] for idx in prediction]
            examples += ' '.join(truth_text) + '/' + ' '.join(preds_text) + '  \n'
        return examples

    def compute_metric(self, captions, preds):
        preds = torch.argmax(preds, axis=2)
        assert preds.shape == captions.shape
        bleu_score = 0.0
        for pred, ground_truth in zip(preds, captions):
            bleu_score += sentence_bleu([ground_truth.tolist()], pred.tolist(), smoothing_function=chencherry.method1)
        return bleu_score/len(captions)