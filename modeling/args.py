import argparse

def parse_args(mode="train"):
    parser = argparse.ArgumentParser()
    
    # basic settings
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
    parser.add_argument("--data_dir",default="./data/train",type=str,help="data directory")
    parser.add_argument("--test_dir",default="./data/eval/asset",type=str,help="data directory")
    parser.add_argument("--model", default="als", type=str, help="model name")
    parser.add_argument("--no_exp_save", action='store_true')  # 해당 인자를 입력할 때만 True로 적용하여 실험 결과를 저장하지 않음
    

    # common settings
    parser.add_argument("--patience", default=3, type=int, help="patient n. for early stopping")
    parser.add_argument("--n_epochs", default=1, type=int, help="iter n")
    
    # als settings
    parser.add_argument("--n_valid", default=1, type=int, help="validation set n")
    parser.add_argument("--n_seq", default=1, type=int, help="sequence n in the validation set")
    parser.add_argument("--factors", default=100, type=int, help="number of factors")
    parser.add_argument("--regularization", default=0.001, type=float, help="regularization")
    parser.add_argument("--als_dir",default="./foodcomImplicit/architects",type=str,help="als architects directory")
    parser.add_argument("--top_k", default=3, type=int, help="recall at k")
    parser.add_argument("--inference_n", default=10, type=int, help="argprtition n for inference")
    
    args = parser.parse_args()

    return args
