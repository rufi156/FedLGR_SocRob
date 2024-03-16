from dataloader.utils import load_datasets_pretrain
from models.deepLabMobileNet import Net as deepNet
from models.MobileNet import Net as mobNet
import torch
import argparse
import pickle
#add utils from previous folder
import sys
sys.path.append('../')
from utils import train, test
def Average(lst):
    return sum(lst) / len(lst)
def run(args):
    #load the datasets
    if args.processor=='cpu':
        DEVICE = torch.device('cpu')
        #add "cpu" to the path
        args.path = args.path + "/cpu"
        gpu = False
    else:
        DEVICE = torch.device('cuda')
        #add "gpu" to the path
        args.path = args.path + "/gpu"
        gpu = True

    if args.models == 'all':
        models=[deepNet(), mobNet()]
        names=['DeepLabMobileNet', 'MobileNet']
    elif args.models == 'DeepLabMobileNet':
        models=[deepNet()]
        names=['DeepLabMobileNet']
    elif args.models == 'MobileNet':
        models=[mobNet()]
        names=['MobileNet']

    data=load_datasets_pretrain(num_clients=1, split= args.split_ratio, batch_size=args.batch_size, path=args.data, aug=False, DEVICE=DEVICE)
    for i in range(len(models)):
        model=models[i]
        model_n=names[i]
        train(model=model, train_loader=data, epochs=args.epochs, DEVICE=DEVICE)
        y_labels=[0,1,2,3,4,5,6,7]
        a,b,c=test(net=model, testloader=data,y_labels=y_labels, DEVICE=DEVICE)
        print(a, b, c)
        with open(args.path + "/" + model_n + ".pkl", 'wb') as f:
            pickle.dump(model.state_dict(), f)
        with open(args.path + "/" + model_n + ".pkl", 'rb') as f:
            model.load_state_dict(pickle.load(f))




if __name__ == "__main__":
    #define parser arguments, we need a model, split_ratio and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, default='all', help='Model to use for training')
    parser.add_argument('-s', '--split_ratio', type=float, default=0.33, help='Split ratio for training')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
    #path to path of data
    parser.add_argument('-d', '--data', type=str, default='SADRA-Dataset', help='Path to data')
    parser.add_argument('-p', '--path', type=str, default='models', help='Path to save the model')
    #processor
    parser.add_argument('-t', '--processor', type=str, default='cpu', help='Processor to run the scrip')
    args = parser.parse_args()

    print("Running with following arguments:")
    print("Training Model: {}".format(args.models))
    print("Split Ratio: {}".format(args.split_ratio))
    print("Batch Size: {}".format(args.batch_size))
    print("Epochs: {}".format(args.epochs))
    print("Path: {}".format(args.path))

    run(args)