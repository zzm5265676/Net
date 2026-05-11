import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.data import get_eval_set, get_SICE_eval_set
from net.CIDNet import CIDNet


def run_eval(model, testing_data_loader, model_path, output_folder,
             norm_size=True, lol=False, v2=False, unpaired=False,
             alpha=1.0, gamma=1.0):
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state)
    print(f'Pre-trained model is loaded from: {model_path}')

    torch.set_grad_enabled(False)
    model.eval()
    print('Evaluation:')

    if lol:
        model.trans.gated = True
    elif v2 or unpaired:
        model.trans.gated2 = True
        model.trans.alpha = alpha

    os.makedirs(output_folder, exist_ok=True)

    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input_tensor, name = batch[0], batch[1]
            else:
                input_tensor, name, h, w = batch[0], batch[1], batch[2], batch[3]

            input_tensor = input_tensor.cuda()
            output = model(input_tensor ** gamma)

        output = torch.clamp(output, 0, 1)
        if not norm_size:
            output = output[:, :, :h, :w]

        output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
        output_img.save(os.path.join(output_folder, name[0]))
        torch.cuda.empty_cache()

    if lol:
        model.trans.gated = False
    elif v2 or unpaired:
        model.trans.gated2 = False

    print('===> End evaluation')
    torch.set_grad_enabled(True)


def build_eval_loader(args):
    norm_size = True
    output_folder = args.output
    alpha = args.alpha

    if args.lol:
        eval_data = DataLoader(
            dataset=get_eval_set(args.input_dir or "./datasets/LOLdataset/eval15/low"),
            num_workers=args.num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = output_folder or './output/LOLv1/'

    elif args.lol_v2_real:
        eval_data = DataLoader(
            dataset=get_eval_set(args.input_dir or "./datasets/LOLv2/Real_captured/Test/Low"),
            num_workers=args.num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = output_folder or './output_kx/LOLv2_real/'

    elif args.lol_v2_syn:
        eval_data = DataLoader(
            dataset=get_eval_set(args.input_dir or "./datasets/LOLv2/Synthetic/Test/Low"),
            num_workers=args.num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = output_folder or './output_kx/LOLv2_syn/'

    elif args.sice_grad:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set(args.input_dir or "./datasets/SICE/SICE_Grad"),
            num_workers=args.num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = output_folder or './output_kx/SICE_grad/'
        norm_size = False

    elif args.sice_mix:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set(args.input_dir or "./datasets/SICE/SICE_Mix"),
            num_workers=args.num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = output_folder or './output_kx/SICE_mix/'
        norm_size = False

    elif args.fivek:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set(args.input_dir or "./datasets/FiveK/test/input"),
            num_workers=args.num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = output_folder or './output_kx/fivek/'
        norm_size = False

    elif args.unpaired:
        eval_data = DataLoader(
            dataset=get_SICE_eval_set(args.input_dir),
            num_workers=args.num_workers,
            batch_size=1,
            shuffle=False,
        )
        output_folder = output_folder or './output_kx/unpaired/'
        norm_size = False

    else:
        raise ValueError("Please specify one dataset flag, e.g. --lol or pass --unpaired with --input_dir.")

    return eval_data, output_folder, norm_size, alpha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval for adaptive k(x) CIDNet')
    parser.add_argument('--weights', type=str, required=True, help='Path to k(x) model weights')
    parser.add_argument('--output', type=str, default=None, help='Output folder')
    parser.add_argument('--input_dir', type=str, default=None, help='Override input directory')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)

    parser.add_argument('--lol', action='store_true')
    parser.add_argument('--lol_v2_real', action='store_true')
    parser.add_argument('--lol_v2_syn', action='store_true')
    parser.add_argument('--sice_grad', action='store_true')
    parser.add_argument('--sice_mix', action='store_true')
    parser.add_argument('--fivek', action='store_true')
    parser.add_argument('--unpaired', action='store_true')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")

    os.makedirs('./output', exist_ok=True)

    eval_data, output_folder, norm_size, alpha = build_eval_loader(args)
    eval_net = CIDNet().cuda()

    run_eval(
        eval_net,
        eval_data,
        args.weights,
        output_folder,
        norm_size=norm_size,
        lol=args.lol,
        v2=args.lol_v2_real,
        unpaired=args.unpaired,
        alpha=alpha,
        gamma=args.gamma,
    )
