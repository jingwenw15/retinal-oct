"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.resnet as resnet
import model.vgg as vgg
import model.net as custom_net
import model.mobilenet as mobilenet
import model.resnet_small as resnet_small
import model.vgg_mse as vgg_mse
import model.resnet_mse as resnet_mse

import model.data_loader as data_loader
from evaluate import evaluate

import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/resnet',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--model', default='vgg')
parser.add_argument('--no_train', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--use_adamw', action='store_true')
parser.add_argument('--use_mse', action='store_true')
parser.add_argument('--wandb_name', default=None)
parser.add_argument('--no_pretrain_weights', action='store_true')

def train(model, optimizer, loss_fn, dataloader, metrics, params, model_name):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    wandb.init(
    # set the wandb project where this run will be logged
    project="cs230",
    name=args.wandb_name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": params.learning_rate,
    "epochs": params.num_epochs,
    "batch_size": params.batch_size,
    "dropout_rate": params.dropout_rate,
    "architecture": model_name,
    "image_size": '64x64',
    "misc": "don't freeze any layers"
    }
    )

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, _) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric + ' train': np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    wandb.log(metrics_mean)
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None, model_name="vgg"):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, model_name)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        wandb.log(val_metrics)

        val_acc = val_metrics['accuracy dev']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

'''
Evaluate the model on the dev and test set. 
'''
def dev_test_model(model, loss_fn, dev_dataloader, test_dataloader, metrics, params, model_name):
    # reload weights from restore_file if specified
    restore_path = os.path.join(
        args.model_dir, 'best.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model, optimizer)
    wandb.init(
    # set the wandb project where this run will be logged
    project="cs230",
    name=args.wandb_name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": params.learning_rate,
    "epochs": params.num_epochs,
    "batch_size": params.batch_size,
    "dropout_rate": params.dropout_rate,
    "architecture": model_name,
    "image_size": '64x64',
    "misc": "dev/test the model"
    },
    )
    dev_metrics = evaluate(model, loss_fn, dev_dataloader, metrics, params, split='dev', write=True, model_name=model_name, run_name=args.wandb_name)
    wandb.log(dev_metrics)
    test_metrics = evaluate(model, loss_fn, test_dataloader, metrics, params, split='test', write=True, model_name=model_name, run_name=args.wandb_name)
    wandb.log(test_metrics)

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    test_dl = None 
    if args.evaluate:
        test_dl = data_loader.fetch_dataloader(['test'], args.data_dir, params)['test']
        logging.info("Loaded test dataset.")

    logging.info("- done.")

    # Define the model and optimizer
    net = None 
    if args.model == "vgg": 
        net = vgg
    elif args.model == "resnet":
        net = resnet
    elif args.model == "resnet_small":
        net = resnet_small
    elif args.model == "custom": 
        net = custom_net
    elif args.model == "mobilenet":
        net = mobilenet
    elif args.model == "vgg_mse": 
        net = vgg_mse
    elif args.model == "resnet_mse": 
        net = resnet_mse
        
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    if args.no_pretrain_weights: model = net.Net(params, False).cuda() if params.cuda else net.Net(params, False)
    
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate) if not args.use_adamw else \
                optim.AdamW(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.ce_loss if not args.use_mse else net.mse_loss
    metrics = net.metrics

    # Train the model
    if not args.no_train: 
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                        args.restore_file, args.model)
    if args.evaluate: 
        dev_test_model(model, loss_fn, val_dl, test_dl, metrics, params, args.model)


