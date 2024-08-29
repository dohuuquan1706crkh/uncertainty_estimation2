import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss.loss as module_loss
import model.metric as module_metric
import model.model.model as module_arch 
from parse_config import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )


    data_loader = config.init_obj('test_data_loader', module_data)
    # print("len test data loader: " ,len(data_loader))
    
    
    
    # print("data loader type" , type(data_loader))
    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    uncertainty_metric_fns = [getattr(module_metric, met) for met in config['uncertainty_metrics']]
    
    
    # load checkpoint of reconstruction model
    logger.info('Loading checkpoint: {} ...'.format(config["resume"]))
    checkpoint = torch.load(config["resume"])
    state_dict = checkpoint['state_dict']
    
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)+len(uncertainty_metric_fns))
    #get frozen model
    frozen_model = config.init_obj('frz_arch', module_arch)
    # frozen_model = frozen_model.to(device)
    for param in frozen_model.parameters():
        param.requires_grad = False
    checkpoint_frz = torch.load(config["frozen_ckpt"])
    state_dict_frz = checkpoint_frz['state_dict']
    frozen_model.load_state_dict(state_dict_frz)
    
    
    with torch.no_grad():
        
        for i, (data, target) in enumerate(tqdm(data_loader)):
            # print(data.shape)
            # data, target = data.to(device), target.to(device)
            y_hat = frozen_model(data)
            uncertainty = model.uncertainty_map(data,y_hat)
            # print(torch.max(uncertainty))
            output = model(data, y_hat)
            predicted_labels = torch.argmax(y_hat, dim=1)  # Shape: [B, W, H]
            # print(predicted_labels.shape)
            # print(target.shape)
            

            # Step 2: Compare with ground truth
            error_map = (predicted_labels != target).float()  # Shape: [B, W, H]
            
            # save sample images, or do something with output here
            if i == 0:                
                error_map_tensor = error_map[3].squeeze(0)
                error_map_np = error_map_tensor.numpy()
                plt.imshow(error_map_np, cmap='gray')  # Use cmap='gray' for grayscale images
                plt.axis('off')  # Optional: Turn off axis labels
                plt.savefig('/raid/quandh/Segmentation-Uncertainty/plot/uncertainty/error_map_image.png', bbox_inches='tight', pad_inches=0)
                # plt.show()
                
                data_tensor = data[3].squeeze(0)
                data_np = data_tensor.numpy()
                plt.imshow(data_np, cmap='gray')  # Use cmap='gray' for grayscale images
                plt.axis('off')  # Optional: Turn off axis labels
                plt.savefig('/raid/quandh/Segmentation-Uncertainty/plot/data_image.png', bbox_inches='tight', pad_inches=0)
                # plt.show()
                
                target_tensor = target[3].squeeze(0)
                colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]  # RGB colors
                target_np = target_tensor.numpy()
                cmap = mcolors.ListedColormap(colors)
                plt.imshow(target_np, cmap=cmap, vmin=0, vmax=3)

                plt.savefig('/raid/quandh/Segmentation-Uncertainty/plot/uncertainty/segmentation_map.png', bbox_inches='tight', pad_inches=0)
                # plt.show()
                
                uncertainty_tensor = uncertainty[3].squeeze(0)
                uncertainty_np = uncertainty_tensor.numpy()
                
                plt.imshow(uncertainty_np, cmap='gray')  # Use cmap='gray' for grayscale images
                plt.axis('off')  # Optional: Turn off axis labels
                plt.savefig('/raid/quandh/Segmentation-Uncertainty/plot/uncertainty/uncertainty_image.png', bbox_inches='tight', pad_inches=0)
                
                output_tensor = output[3].squeeze(0)
                
                output_tensor = torch.argmax(output_tensor, dim=0)  # Resulting shape is (256, 256)
                output_np = output_tensor.numpy()
                plt.imshow(output_np, cmap=cmap, vmin=0, vmax=3)

                plt.axis('off')  # Optional: Turn off axis labels
                plt.savefig('/raid/quandh/Segmentation-Uncertainty/plot/uncertainty/output_image.png', bbox_inches='tight', pad_inches=0)
                # plt.show()
                # Plot the histogram
                
                uncertainty_tensor = uncertainty_tensor.flatten()
                # print(torch.max(output_tensor))
                plt.figure(figsize=(8, 6))
                plt.hist(uncertainty_tensor.numpy(), bins=100, color='blue', edgecolor='black')

                # Add title and labels
                # plt.title('Histogram of Tensor Values')
                plt.xlabel('Value')
                plt.ylabel('Frequency')

                # Save the plot as an image file
                plt.savefig('/raid/quandh/Segmentation-Uncertainty/plot/uncertainty/output_histogram.png')


            # computing loss, metrics on test set
            loss = loss_fn(y_hat, output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            for i, metric in enumerate(uncertainty_metric_fns):
                total_metrics[i+len(metric_fns)] += metric(y_hat, target, uncertainty) * batch_size

    n_samples = len(data_loader.sampler)
    # print("test sample")
    # print(n_samples)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log.update({
        met.__name__: total_metrics[i+len(metric_fns)].item() / n_samples for i, met in enumerate(uncertainty_metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
