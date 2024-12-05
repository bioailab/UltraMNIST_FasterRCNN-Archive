import os
import argparse
from ultramnist import CreateUltraMNIST

ap = argparse.ArgumentParser()
ap.add_argument("--root_path", type=str, default="/data/", help="path to the root directory where the generated data will be stored. This is relative to the current directory.")
ap.add_argument("--n_samples", type=int, default=28, help="number of train and test samples")

args = ap.parse_args()
print(os.getcwd())
print(os.getcwd() + args.root_path + f'ultramnist_{args.n_samples}')
obj_umnist = CreateUltraMNIST(root= os.getcwd() + args.root_path + f'ultramnist_{args.n_samples}', 
                                base_data_path=os.path.join(args.root_path, 'mnist'), 
                                n_samples = [args.n_samples, args.n_samples], 
                                img_size=4000) # 4000
obj_umnist.generate_dataset()