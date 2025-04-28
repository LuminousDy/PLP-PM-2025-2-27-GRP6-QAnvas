import argparse

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for HuggingFace TrainingArguments")

    # === Output and Logging ===
    parser.add_argument('--output_dir', type=str, default="/content/drive/MyDrive/EBA5004/t5-multi-intentv2",
                        help="Directory to save checkpoints and model")
    parser.add_argument('--logging_dir', type=str, default="./logs", help="Directory for logs")


    # === Training Parameters ===
    parser.add_argument('--train_batch_size', type=int, default=4, help="Batch size per device for training")
    parser.add_argument('--eval_batch_size', type=int, default=4, help="Batch size per device for evaluation")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=10, help="Number of warmup steps")
    parser.add_argument('--logging_steps', type=int, default=10, help="Logging frequency in steps")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument('--train_dataset', type=str, default="/content/drive/MyDrive/EBA5004/multi_intent_dataset_augmented_v2.json",
                        help="Path to the training dataset")
    
    # === Strategy and Mixed Precision ===
    parser.add_argument('--save_strategy', type=str, default="epoch", help="Save checkpoint strategy (e.g., epoch, steps)")
    parser.add_argument('--evaluation_strategy', type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument('--fp16', action='store_true', help="Use mixed precision (FP16)")
    parser.add_argument('--report_to', type=str, default="none", help="Reporting tool (e.g., wandb, none)")

    # === Inference Parameters ===
    parser.add_argument('--max_length', type=int, default=128, help="Maximum length of the generated sequence") 
    parser.add_checkpoints('--model_path', type=str, default="/media/labpc2x2080ti/data/Mohan_Workspace/checkpoint-225000",)
    return parser.parse_args()

