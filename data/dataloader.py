import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import AutoTokenizer
import pandas as pd

class kashmirDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name='bert-base-uncased', max_length_tokens=512, max_length_audio=16000, transform=None):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length_tokens = max_length_tokens
        self.max_length_audio = max_length_audio
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.dataframe.iloc[idx]['filename_segment']
        audio_file_path = self.dataframe.iloc[idx]['path']
 
        waveform, _ = torchaudio.load(audio_file_path)
        waveform = self.cut_if_necessary(waveform)
        waveform = self.right_pad_if_necessary(waveform)
        waveform = waveform.unsqueeze(1)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        waveform = waveform.squeeze(1)

  
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length_tokens, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()


        target_ids = self.create_target_tokens(input_ids)

 
        data = {
            'input_tokens': input_ids,
            'speech_features': waveform,
            'attentionmask': attention_mask,
            'target_tokens': target_ids
        }
        
        return data

    def cut_if_necessary(self, waveform):
        if waveform.shape[1] > self.max_length_audio:
            waveform = waveform[:, :self.max_length_audio]
        return waveform

    def right_pad_if_necessary(self, waveform):
        length = waveform.shape[1]
        if self.max_length_audio > length:
            pad_last_dim = (0, self.max_length_audio - length)
            waveform = torch.nn.functional.pad(waveform, pad_last_dim)
        return waveform

    def create_target_tokens(self, input_ids):

        eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id  
        target_ids = input_ids[1:]  
        target_ids = torch.cat([target_ids, torch.tensor([eos_token_id])], dim=0) 
        if len(target_ids) > self.max_length_tokens:
            target_ids = target_ids[:self.max_length_tokens]
        return target_ids

def collate_fn(batch):

    input_tokens = torch.stack([item['input_tokens'] for item in batch])
    speech_features = torch.stack([item['speech_features'] for item in batch])
    attentionmask = torch.stack([item['attentionmask'] for item in batch])
    target_tokens = torch.stack([item['target_tokens'] for item in batch])
    
    data = {
        'input_tokens': input_tokens,
        'speech_features': speech_features,
        'attentionmask': attentionmask,
        'target_tokens': target_tokens
    }
    
    return data
