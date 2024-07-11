import argparse
from transformers import AutoTokenizer

from utils import *
from train_utils import *
from model import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "Hello, I am Pritam. Today when I was completing my assignment, I heard a loud noise."

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "LoRA":    
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        # TODO: Also plot the training losses and metrics
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        print("LoRA training starting...")
        print(f"Using Adam optimizer with learning rate {args.lr}.")
        train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs)
        print("LoRA training completed sucessful")

        model.save_trainable_params(args.model_path)
        
    elif args.mode == "distil":
        teacher_model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        teacher_model.load_trainable_params(args.model_path)
        teacher_model.eval()

        student_model = DistilRNN(hidden_dim=args.hidden_dim).to(args.device)  # TODO: Implement the student model class
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        # HINT: You can use an additional parameter in train function to differentiate LoRA and distillation training, no changes in evaluate function required.
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
        distil_criterion = torch.nn.KLDivLoss()
        criterion = torch.nn.CrossEntropyLoss()
        print("distil starting...")
        print(f"Using Adam optimizer with learning rate {args.lr}.")
        train_distil(teacher_model, student_model, train_loader, val_loader, optimizer, criterion, distil_criterion, args.device, args.epochs)
        print("distil done")
        # raise NotImplementedError

    elif args.mode == "rnn":
        model = DistilRNN().to(args.device)
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        print("RNN starting...")
        # print(f"Initialized RNN model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
        print(f"Using Adam optimizer with learning rate {args.lr}.")
        train(model, train_loader, val_loader, optimizer, criterion, args.device, args.epochs, is_rnn=True)
        print("RNN done")
        # raise NotImplementedError
    else:
        print("Invalid mode")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn"], help="Mode to run the program in")
    parser.add_argument("sr_no", type=int, help="5 digit SR number")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN Hidden Dimensions")
    # TODO: Add more arguments as needed
    # parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of data reserved for validation")
    # parser.add_argument("--early_stopping", type=int, default=0, help="Number of epochs to wait before early stop if no progress on the validation set")
    # parser.add_argument("--verbose", action='store_true', help="Increase output verbosity")

    args = parser.parse_args()
    args.device = torch.device(
        "cuda:4" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    seed_everything(args.sr_no)

    main(args)
