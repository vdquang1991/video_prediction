import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from args import get_parser
from data import get_data, clean_data, VideoDataset
from train import get_models

def generate_step(x, models, opt, device):
    frame_predictor = models['frame_predictor']
    posterior = models['posterior']
    prior = models['prior']
    encoder = models['encoder']
    decoder = models['decoder']

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden(device)
    posterior.hidden = posterior.init_hidden(device)
    prior.hidden = prior.init_hidden(device)

    gen_seq = []
    with torch.no_grad():
        x_in = x[0]
        gen_seq.append(x_in.data.cpu().numpy())
        for i in range(1, opt.n_past + opt.n_eval):
            h, skip = encoder(x_in)
            h = h.detach()
            if i == 1:
                skips = skip
            else:
                skips = [(skips[idx] * (i - 1) + skip[idx]) / i for idx in range(len(skips))]
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, mu, logvar = posterior(h_target)
            else:
                z_t, _, _ = prior(h)
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skips])
            if i < opt.n_past:
                x_in = x[i]
            else:
                x_in = x_pred
            gen_seq.append(x_in.data.cpu().numpy())
    return np.asarray(gen_seq)

def generate(opt, dataset_root, test_dataset, save_path='./save_path', test_steps=1000, device='cpu'):
    # Create root folder for predicted video
    predict_root_path = os.path.join(dataset_root, 'predict_video')
    if not os.path.exists(predict_root_path):
        os.mkdir(predict_root_path)
    # ---------------- Create the models  ----------------
    models, optims = get_models(opt, device)
    # ---------------- Load the models weights ----------------
    print('Load model weigts at: %s/best_ssim_model.pth' % (save_path))
    if device=='cpu':
        tmp = torch.load('%s/best_ssim_model.pth' % (save_path), map_location=torch.device('cpu'))
    else:
        tmp = torch.load('%s/best_ssim_model.pth' % (save_path))
    models['frame_predictor'] = tmp['frame_predictor']
    models['posterior'] = tmp['posterior']
    models['prior'] = tmp['prior']
    models['encoder'] = tmp['encoder']
    models['decoder'] = tmp['decoder']
    print('Load model weights completed')

    models['frame_predictor'].eval()
    models['posterior'].eval()
    models['prior'].eval()
    models['encoder'].eval()
    models['decoder'].eval()

    # --------- load a dataset ------------------------------------
    test_data = VideoDataset(root_path=dataset_root, data=test_dataset, n_past=opt.n_past, n_future=opt.n_eval,
                             image_size=opt.image_size, n_channel=opt.n_channel, is_train=False)
    test_loader = DataLoader(test_data,
                             num_workers=8,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    def get_testing_batch():
        while True:
            for sequence in test_loader:
                clip, row, start_idx = sequence
                yield (clip.permute(1, 0, 4, 2, 3).to(device), row, start_idx)
    testing_batch_generator = get_testing_batch()

    # --------- run generate ------------------------------------
    for i in range(test_steps):
        x, row, start_idx = next(testing_batch_generator)
        x_pred = generate_step(x, models, opt, device)
        x_pred = np.transpose(x_pred, axes=(1, 0, 3, 4, 2))
        for batch in range(opt.batch_size):
            x_pred_batch = x_pred[batch]
            x_pred_batch *= 255.
            x_pred_batch = np.asarray(x_pred_batch, dtype=np.uint8)
            if opt.n_channel == 1:
                x_pred_batch = np.repeat(x_pred_batch, 3, axis=-1)
            folder_name = row[1][batch]
            video_name = row[2][batch]
            if not os.path.exists(os.path.join(predict_root_path, folder_name)):
                os.mkdir(os.path.join(predict_root_path, folder_name))
            predicted_video_path = os.path.join(predict_root_path, folder_name, video_name)
            if not os.path.exists(predicted_video_path):
                os.mkdir(predicted_video_path)
                print('---->Processing at video name: ', predicted_video_path)
            for j in range(opt.n_past + opt.n_eval):
                img_path = os.path.join(predicted_video_path, '%05d.jpg' % (start_idx[batch] + j))
                cv2.imwrite(img_path, x_pred_batch[j,:,:,::-1])
    print('Prediction completed!')

def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)  # Choose GPU for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root = opt.dataset.upper()
    # Create the folder to save model and checkpoint
    save_path = os.path.join(dataset_root, 'save_models')

    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD

    # Read dataset
    test_dataset = get_data(os.path.join(dataset_root, 'test.csv'))
    print('Test set:', len(test_dataset))
    test_dataset = clean_data(test_dataset, opt.n_past + opt.n_eval, MAX_FRAMES=3000)
    print('Test set after clean:', len(test_dataset))

    generate(opt, dataset_root=dataset_root, test_dataset=test_dataset, save_path=save_path, test_steps=len(test_dataset), device=device)


if __name__ == '__main__':
    opt = get_parser()
    main(opt)