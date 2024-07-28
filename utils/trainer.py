from decode import decode_predictions,calculate_wer,evaluate_and_write_to_csv

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import logging

class Trainer():
    def __init__(self, model, train_dataloader, test_dataloader, tokenizer,lr, wt_decay, epochs, batch_size, log_dir,beam_size):
        self.model = model.to('cuda')
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wt_decay)
        self.criterion = nn.CrossEntropyLoss(ignore_index=100)  # pad token
        self.lr = lr
        self.wt_decay = wt_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.total_wer = None

        # Setup logging
        logging.basicConfig(filename=f'training_log_{self.model.tencoder}_{self.model.text_decoder}.txt', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

    def train_step(self, mel_inp_ids, tgt_ids, padding_mask, labels):
        self.model.train()
        mel_inp_ids = mel_inp_ids.to('cuda')
        tgt_ids = tgt_ids.to('cuda')
        padding_mask = padding_mask.to('cuda')
        labels = labels.to('cuda')

        self.optimizer.zero_grad()
        outs = self.model(mel_inp_ids, tgt_ids, padding_mask)
        loss = self.criterion(outs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        total_wer = 0.0
        with torch.no_grad():
            for batch in self.test_dataloader:
                mel_inp_ids, tgt_ids, padding_mask, labels = batch
                mel_inp_ids = mel_inp_ids.to('cuda')
                tgt_ids = tgt_ids.to('cuda')
                padding_mask = padding_mask.to('cuda')
                labels = labels.to('cuda')

                outs = self.model(mel_inp_ids, tgt_ids, padding_mask)
                loss = self.criterion(outs, labels)
                val_loss += loss.item()

                pred_texts = decode_predictions(outs)
                target_texts = decode_predictions(labels)

                batch_wer = calculate_wer(pred_texts, target_texts)
                total_wer += batch_wer
        
        avg_val_loss = val_loss / len(self.test_dataloader)
        avg_wer = total_wer / len(self.test_dataloader)
        return avg_val_loss, avg_wer

    def train(self):
        for epoch in range(self.epochs):
            train_loss = 0.0
            for batch in self.train_dataloader:
                mel_inp_ids, tgt_ids, padding_mask, labels = batch
                loss = self.train_step(mel_inp_ids, tgt_ids, padding_mask, labels)
                train_loss += loss
                self.global_step += 1
                self.writer.add_scalar('Train/Loss', loss, self.global_step)

            avg_train_loss = train_loss / len(self.train_dataloader)
            avg_val_loss, avg_wer = self.validate()

            self.writer.add_scalar('Train/Average_Loss', avg_train_loss, epoch)
            self.writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
            self.writer.add_scalar('Validation/WER', avg_wer, epoch)

            log_message = f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation WER: {avg_wer:.4f}"
            self.total_wer = avg_wer
            self.log(log_message)
        torch.save(obj=self.model.state_dict(),
           f=f'/content/outs/{self.model.tencoder}_{self.model.text_decoder}_{self.total_wer}')
    
    def evaluate(self):
        model_e = self.model
        model_e = model_e.load_state_dict(torch.load(f=f'/content/outs/{self.model.tencoder}_{self.model.text_decoder}_{self.total_wer}'))
        self.log('Getting predictions')
        evaluate_and_write_to_csv(self.test_dataloader,model_e,self.tokenizer,self.beam_size,f'log_dir/{self.model.tencoder}_{self.model.text_decoder}')
        
    def log(self, message):
        self.logger.info(message)

