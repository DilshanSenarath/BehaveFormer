import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--mode")
parser.add_argument("--dataset")
parser.add_argument("--metric")
parser.add_argument("--testfile")
parser.add_argument("--epochs")
parser.add_argument("--initepoch")
parser.add_argument("--imu")
args = parser.parse_args()

VALID_ARG_VALUES = {
    "model": ["keystroke", "keystroke_imu", "tl", None],
    "mode": ["preprocess", "train", "continue_train", "test", None],
    "dataset": ["aalto", "hmog", "humi", None],
    "metric": ["basic", "det", "pca", None],
    "imu": ["acc", "gyr", "mag", "acc_gyr", "acc_mag", "mag_gyr", "all", None]
}

Map = {
    "keystroke": (Path(__file__)/"../").resolve()/Path("experiments/keystroke"),
    "keystroke_imu": (Path(__file__)/"../").resolve()/Path("experiments/keystroke_imu_combined"),
    "tl": (Path(__file__)/"../").resolve()/Path("experiments/transfer_learning"),
    "aalto": "AaltoDB",
    "hmog": "HMOGDB",
    "humi": "HuMIdb"
}

def validation(args):
    if (args.model not in VALID_ARG_VALUES["model"]):
        return False
    elif (args.mode not in VALID_ARG_VALUES["mode"]):
        return False
    elif (args.dataset not in VALID_ARG_VALUES["dataset"]):
        return False
    elif (args.metric not in VALID_ARG_VALUES["metric"]):
        return False
    elif (args.imu not in VALID_ARG_VALUES["imu"]):
        return False
    return True

if __name__ == "__main__":
    if (validation(args)):
        if (args.model != None):
            if (args.model == "tl"):
                if (args.mode == "test"):
                    subprocess.run(f"python {Map[args.model]/args.mode}.py {args.metric} {args.testfile}", shell=True)
                elif (args.mode == "train"):
                    subprocess.run(f"python {Map[args.model]/f'{args.mode}_{Map[args.dataset]}'}.py {args.epochs}", shell=True)
                elif (args.mode == "continue_train"):
                    subprocess.run(f"python {Map[args.model]/f'train_{Map[args.dataset]}'}.py {args.epochs} {args.initepoch}", shell=True)
            else:
                if (args.imu != None):
                    if (args.mode == "test"):
                        subprocess.run(f"python {Map[args.model]/Map[args.dataset]/f'imu_{args.imu}'/args.mode}.py {args.metric} {args.testfile}", shell=True)
                    elif (args.mode == "train"):
                        subprocess.run(f"python {Map[args.model]/Map[args.dataset]/f'imu_{args.imu}'/args.mode}.py {args.epochs}", shell=True)
                    elif (args.mode == "continue_train"):
                        subprocess.run(f"python {Map[args.model]/Map[args.dataset]/f'imu_{args.imu}'}/train.py {args.epochs} {args.initepoch}", shell=True)
                else:
                    if (args.mode == "test"):
                        subprocess.run(f"python {Map[args.model]/Map[args.dataset]/args.mode}.py {args.metric} {args.testfile}", shell=True)
                    elif (args.mode == "train"):
                        subprocess.run(f"python {Map[args.model]/Map[args.dataset]/args.mode}.py {args.epochs}", shell=True)
                    elif (args.mode == "continue_train"):
                        subprocess.run(f"python {Map[args.model]/Map[args.dataset]}/train.py {args.epochs} {args.initepoch}", shell=True)
        else:
            subprocess.run(f"python {str((Path(__file__)/'../').resolve())}/data/{Map[args.dataset]}/{args.mode}.py", shell=True)
    else:
        raise ValueError("Please give correct values for arguments")
