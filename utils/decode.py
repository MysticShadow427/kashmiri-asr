# write a function to write a csv file to write al the pred and the wers also these funcs - decode_predictions,calculate_wer
import torch
import torch.nn.functional as F
from typing import List, Tuple
import jiwer
import csv

def beam_search_decoder(predictions: torch.Tensor, beam_size: int) -> List[List[int]]:
    batch_size, seq_len, vocab_size = predictions.size()
    sequences = [[([], 0.0)] for _ in range(batch_size)]
    
    for t in range(seq_len):
        for i in range(batch_size):
            all_candidates = []
            for seq, score in sequences[i]:
                prob = F.log_softmax(predictions[i, t], dim=-1)
                topk_probs, topk_indices = torch.topk(prob, beam_size)
                for k in range(beam_size):
                    candidate = (seq + [topk_indices[k].item()], score - topk_probs[k].item())
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda x: x[1])
            sequences[i] = ordered[:beam_size]
    
    return [seq[0][0] for seq in sequences]

def decode_predictions(sequences: List[List[int]], tokenizer) -> List[str]:
    return [tokenizer.decode(seq) for seq in sequences]

def calculate_wer(ref: str, hyp: str) -> float:
    return jiwer.wer(ref, hyp)

def evaluate_and_write_to_csv(dataloader, model, tokenizer, beam_size: int, output_csv: str):
    all_references = []
    all_hypotheses = []
    all_wers = []

    model = model.to('cuda')
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            mel_inp_ids, tgt_ids, padding_mask, labels = batch
            mel_inp_ids = mel_inp_ids.to('cuda')
            tgt_ids = tgt_ids.to('cuda')
            padding_mask = padding_mask.to('cuda')
            labels = labels.to('cuda')

            logits = model(mel_inp_ids, tgt_ids, padding_mask)
            decoded_sequences = beam_search_decoder(logits, beam_size)
            decoded_sentences = decode_predictions(decoded_sequences, tokenizer)
            references = decode_predictions(tgt_ids)
            for ref, hyp in zip(references, decoded_sentences):
                wer = calculate_wer(ref, hyp)
                all_references.append(ref)
                all_hypotheses.append(hyp)
                all_wers.append(wer)
    
    overall_wer = sum(all_wers) / len(all_wers)
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Reference', 'Hypothesis', 'WER'])
        for ref, hyp, wer in zip(all_references, all_hypotheses, all_wers):
            writer.writerow([ref, hyp, wer])
        writer.writerow(['Overall WER', '', overall_wer])
    
    print(f'Results written to {output_csv}')
    print(f'Overall WER: {overall_wer}')