import torch
from transformers import AutoTokenizer
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import BERTScore

class MetricsTest:
    def __init__(self):
        # Initialize the tokenizer (using a small T5 model for this example)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        # Initialize metrics
        self.rouge_score = ROUGEScore(use_stemmer=True)
        self.bert_score = BERTScore(model_name_or_path="microsoft/deberta-large-mnli")
        self.bleu_score = BLEUScore()

    def _compute_metrics(self, predictions, labels):
        decoded_preds = predictions      
        decoded_labels = labels
        
        # Calculate ROUGE scores
        result_rouge = self.rouge_score(preds=decoded_preds, target=decoded_labels)
        
        # Calculate BERT scores
        result_brt = self.bert_score(preds=decoded_preds, target=decoded_labels)
        result_brt_average_values = {key: torch.tensor(tensors.mean().item()) for key, tensors in result_brt.items()}
        
        # Calculate BLEU score - Fixed the data structure
        tokenized_preds = [pred.split() for pred in decoded_preds]
        tokenized_labels = [[ref.split()] for ref in decoded_labels]  # Note: BLEU expects list of lists of references
        
        # The correct format for BLEU:
        # preds = [['token1', 'token2', ...], ['token1', 'token2', ...], ...]
        # target = [[['token1', 'token2', ...]], [['token1', 'token2', ...]], ...]
        bleu_score = self.bleu_score(tokenized_preds, tokenized_labels)
        
        results = {**result_rouge, **result_brt_average_values, 'bleu_score': bleu_score}
        return results

def main():
    # Create some example translations
    predictions = [
        "The cat is sitting on the mat.",
        "I love to eat pizza for dinner.",
        "The weather is very nice today."
    ]
    
    ground_truth = [
        "A cat sits on the mat.",
        "I really enjoy eating pizza for dinner.",
        "Today's weather is beautiful."
    ]
    
    # Initialize the test class
    metrics_test = MetricsTest()
    
    # Calculate metrics
    results = metrics_test._compute_metrics(predictions, ground_truth)
    
    # Print results in a formatted way
    print("\nMetrics Results:")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric:25}: {value:.4f}")

if __name__ == "__main__":
    main()