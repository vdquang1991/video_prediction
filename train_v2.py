import os
import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import progressbar
import time
from args import get_parser
from data import get_data, clean_data, VideoDataset
from models.model_v2 import Encoder, Decoder, conv_lstm, gaussian_conv_lstm
import utils
from metrics.lpips.loss import PerceptualLoss

def get_optimizers(models, opt):
    return {key: opt.optimizer(val.parameters(), lr=opt.lr, betas=(0.9, 0.999)) for key, val in models.items()}

def get_models(opt, device):
    models = {}
    # --------- build models ------------------------------------
    print('Skip_type:', opt.skip_type)
    encoder = Encoder(opt.g_dim, init_filters=opt.init_filters, n_channel=opt.n_channel, activation_type=opt.act)
    decoder = Decoder(opt.g_dim, init_filters=opt.init_filters, n_channel=opt.n_channel, skip_type=opt.skip_type, activation_type=opt.act)
    img_dim = opt.image_size // 4
    frame_predictor = conv_lstm(opt.g_dim * 2 + opt.z_dim, opt.g_dim * 2, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size, (img_dim, img_dim), device)
    posterior = gaussian_conv_lstm(opt.g_dim * 2, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size, (img_dim, img_dim), device)
    prior = gaussian_conv_lstm(opt.g_dim * 2, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size, (img_dim, img_dim), device)

    # init weights
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

    # --------- transfer to gpu ------------------------------------
    models['frame_predictor'] = frame_predictor.to(device)
    models['posterior'] = posterior.to(device)
    models['prior'] = prior.to(device)
    models['encoder'] = encoder.to(device)
    models['decoder'] = decoder.to(device)

    optimizers = get_optimizers(models, opt)
    return models, optimizers

# --------- training funtions ------------------------------------
def train_step(x, models, optims, mse_criterion, l1_criterion, kl_criterion, gdl_criterion, bce_criterion, opt, device):
    frame_predictor = models['frame_predictor']
    posterior = models['posterior']
    prior = models['prior']
    encoder = models['encoder']
    decoder = models['decoder']
    frame_predictor.zero_grad()
    prior.zero_grad()
    posterior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden(device)
    posterior.hidden = posterior.init_hidden(device)
    prior.hidden = prior.init_hidden(device)

    mse = 0.
    l1 = 0.
    kld = 0.
    gdl = 0.
    bce = 0.
    N = opt.n_past + opt.n_future
    x_rgb = x[1]
    x_diff = x_rgb - x[0]
    x_diff[x_diff < opt.c] = 0.
    x_diff[x_diff >= opt.c] = 1.
    for i in range(2, N):
        h, skip = encoder(x_rgb, x_diff)
        if i == 2:
            skips = skip
        else:
            skips = [(skips[idx] * (i - 2) + skip[idx]) / (i - 1) for idx in range(len(skips))]
        x_diff_target = x[i] - x_rgb
        x_diff_target[x_diff_target < opt.c] = 0.
        x_diff_target[x_diff_target >= opt.c] = 1.
        h_target = encoder(x[i], x_diff_target)[0]
        z_t, mu, logvar = posterior(h_target)
        _, mu_p, logvar_p = prior(h)
        h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred_rgb, x_pred_diff = decoder([h_pred, skips])
        mse += mse_criterion(x_pred_rgb, x[i])
        l1 += l1_criterion(x_pred_rgb, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p, opt.batch_size)
        bce += bce_criterion(x_pred_diff, x_diff_target)
        if opt.n_channel == 1:
            gdl += gdl_criterion(x_pred_rgb.repeat_interleave(3, dim=1), x[i].repeat_interleave(3, dim=1))
        else:
            gdl += gdl_criterion(x_pred_rgb, x[i])
        # Prepare for next frame
        x_diff = x_diff_target
        x_rgb = x[i]
        if (i >= opt.n_past) and (opt.sch_sampling != 0) and (random.random() > opt.sc_prob):
            x_diff = x_pred_diff
        if (i >= opt.n_past) and (opt.sch_sampling != 0) and (random.random() > opt.sc_prob):
            x_rgb = x_pred_rgb

    loss = mse + l1 + opt.beta_gdl * gdl + opt.beta_bce * bce + opt.beta_kld * kld
    loss.backward()
    for optimizer in optims.values():
        optimizer.step()
    mse = mse.data.cpu().numpy() / N
    l1 = l1.data.cpu().numpy() / N
    bce = bce.data.cpu().numpy() / N
    kld = kld.data.cpu().numpy() / N
    gdl = gdl.data.cpu().numpy() / N
    loss = loss.data.cpu().numpy() / N
    return loss, mse, gdl, bce, l1, kld

def eval_step(x, models, opt, device, lpips_model):
    frame_predictor = models['frame_predictor']
    posterior = models['posterior']
    prior = models['prior']
    encoder = models['encoder']
    decoder = models['decoder']

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden(device)
    posterior.hidden = posterior.init_hidden(device)
    prior.hidden = prior.init_hidden(device)

    with torch.no_grad():
        gen_seq = []
        gt_seq = []
        x_rgb = x[1]
        x_diff = x_rgb - x[0]
        x_diff[x_diff < opt.c] = 0.
        x_diff[x_diff >= opt.c] = 1.
        for i in range(2, opt.n_past + opt.n_eval):
            h, skip = encoder(x_rgb, x_diff)
            h = h.detach()
            if i == 2:
                skips = skip
            else:
                skips = [(skips[idx] * (i - 2) + skip[idx]) / (i - 1) for idx in range(len(skips))]
            if i < opt.n_past:
                x_diff_target = x[i] - x_rgb
                x_diff_target[x_diff_target < opt.c] = 0.
                x_diff_target[x_diff_target >= opt.c] = 1.
                h_target = encoder(x[i], x_diff_target)[0]
                h_target = h_target.detach()
                z_t, mu, logvar = posterior(h_target)
            else:
                z_t, _, _ = prior(h)
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred_rgb, x_pred_diff = decoder([h_pred, skips])
            if i < opt.n_past:
                x_diff = x_diff_target
                x_rgb = x[i]
            else:
                x_diff = x_pred_diff
                x_rgb = x_pred_rgb
                gen_seq.append(x_pred_rgb.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())

        gen_seq = torch.from_numpy(np.stack(gen_seq))
        gt_seq = torch.from_numpy(np.stack(gt_seq))
        ssim_score = utils._ssim_wrapper(gen_seq, gt_seq).mean(2).mean(0)
        mse = torch.mean(F.mse_loss(gen_seq, gt_seq, reduction='none'), dim=[3, 4])
        pnsr_score = 10 * torch.log10(1 / mse).mean(2).mean(0).cpu()
        lpips_score = utils._lpips_wrapper(gen_seq, gt_seq, lpips_model).mean(0).cpu()
        results = {
            'mse': mse,
            'psnr': pnsr_score,
            'ssim': ssim_score,
            'lpips': lpips_score
        }
        return results

def training(opt, dataset_root, train_dataset, test_dataset, save_path='./save_path', training_steps=1000, test_steps=500, device='cpu'):
    # ---------------- Create the models  ----------------
    models, optims = get_models(opt, device)

    all_metrics = {'train_loss': [], 'train_mse_loss': [], 'train_bce_loss': [], 'train_gdl_loss': [], 'train_l1_loss': [], 'train_kld_loss': [],
                   'psnr': [], 'ssim': [], 'lpips': []}
    best_psnr_value = 0.
    best_ssim_value = 0.
    # ---------------- Load the model weights and continuous training if start epoch > 1 ----------------
    if opt.start_epoch > 1:
        print('Load model weights and continuous training')
        checkpoints_path = os.path.join(save_path, 'checkpoints')
        tmp = torch.load('%s/epoch_%03d_model.pth' % (checkpoints_path, opt.start_epoch))
        models['frame_predictor'] = tmp['frame_predictor']
        models['posterior'] = tmp['posterior']
        models['prior'] = tmp['prior']
        models['encoder'] = tmp['encoder']
        models['decoder'] = tmp['decoder']
        jsonfile = os.path.join(save_path, 'checkpoints', 'log.json')
        all_metrics = json.loads(open(jsonfile).read())
        for k in all_metrics.keys():
            all_metrics[k] = all_metrics[k][:opt.start_epoch]
        best_psnr_value = max(all_metrics['psnr'])
        best_ssim_value = max(all_metrics['ssim'])
        print('Current best psnr: ', best_psnr_value)
        print('Current best ssim: ', best_ssim_value)

    # --------- loss functions ------------------------------------
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    gdl_criterion = utils.GDL()
    l1_criterion = nn.L1Loss()
    kl_criterion = utils.kl_criterion

    # --------- load a dataset ------------------------------------
    train_data = VideoDataset(root_path=dataset_root, data=train_dataset, n_past=opt.n_past, n_future=opt.n_future,
                              image_size=opt.image_size, n_channel=opt.n_channel, is_train=True)
    test_data = VideoDataset(root_path=dataset_root, data=test_dataset, n_past=opt.n_past, n_future=opt.n_eval,
                              image_size=opt.image_size, n_channel=opt.n_channel, is_train=False)

    train_loader = DataLoader(train_data,
                              num_workers=8,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=8,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    def get_training_batch():
        while True:
            for sequence in train_loader:
                yield sequence.permute(1, 0, 4, 2, 3).to(device)
    def get_testing_batch():
        while True:
            for sequence in test_loader:
                sequence = sequence[0]
                yield sequence.permute(1, 0, 4, 2, 3).to(device)
    training_batch_generator = get_training_batch()
    testing_batch_generator = get_testing_batch()
    # Load model for LPIPS metrics
    if device == 'cpu':
        lpips_model = PerceptualLoss('lpips_weights', use_gpu=False)
    else:
        lpips_model = PerceptualLoss('lpips_weights', use_gpu=True)

    init_lr = opt.lr
    for epoch in range(opt.start_epoch, opt.epochs):
        print('\n')
        # --------- training loop ------------------------------------
        models['frame_predictor'].train()
        models['posterior'].train()
        models['prior'].train()
        models['encoder'].train()
        models['decoder'].train()

        epoch_loss = 0.
        epoch_mse = 0.
        epoch_gdl_loss = 0.
        epoch_bce_loss = 0.
        epoch_l1_loss = 0.
        epoch_kld_loss = 0.

        # Reduce lr
        if epoch > opt.epochs / 3.:
            print('current lr = %f: ' % (init_lr * 0.1))
            for key in optims.keys():
                optims[key].param_groups[0]['lr'] = init_lr * 0.1

        if epoch > opt.epochs * 2 / 3.:
            print('current lr = %f: ' % (init_lr * 0.01))
            for key in optims.keys():
                optims[key].param_groups[0]['lr'] = init_lr * 0.01

        # Cal sch_sampling
        if opt.sch_sampling != 0:
            opt.sc_prob = opt.sch_sampling / (opt.sch_sampling + np.exp(epoch / opt.sch_sampling))
            print(opt.sc_prob)
        else:
            opt.sc_prob = 1.

        progress = progressbar.ProgressBar(max_value=training_steps).start()
        start_time = time.time()
        for i in range(training_steps):
            progress.update(i + 1)
            x = next(training_batch_generator)
            loss, mse, gdl, bce, l1, kld = train_step(x, models, optims, mse_criterion, l1_criterion, kl_criterion, gdl_criterion, bce_criterion, opt, device)
            epoch_loss += loss
            epoch_mse += mse
            epoch_gdl_loss += gdl
            epoch_bce_loss += bce
            epoch_l1_loss += l1
            epoch_kld_loss += kld

        progress.finish()
        utils.clear_progressbar()
        end_time = time.time()

        print('Epoch [%03d]: train loss: %.5f | mse loss: %.5f | gdl loss: %.5f | bce loss: %.5f | l1 loss: %.5f | kld loss: %.5f | training time: %.3f (%d)' % (
        epoch, epoch_loss / training_steps, epoch_mse / training_steps, epoch_gdl_loss /training_steps, epoch_bce_loss / training_steps, epoch_l1_loss / training_steps, epoch_kld_loss / training_steps,
        end_time-start_time, epoch * training_steps * opt.batch_size))
        all_metrics['train_loss'].append(epoch_loss / training_steps)
        all_metrics['train_mse_loss'].append(epoch_mse / training_steps)
        all_metrics['train_bce_loss'].append(epoch_bce_loss / training_steps)
        all_metrics['train_gdl_loss'].append(epoch_gdl_loss / training_steps)
        all_metrics['train_kld_loss'].append(epoch_kld_loss / training_steps)
        all_metrics['train_l1_loss'].append(epoch_l1_loss / training_steps)


        # --------- run eval ------------------------------------
        models['frame_predictor'].eval()
        models['posterior'].eval()
        models['prior'].eval()
        models['encoder'].eval()
        models['decoder'].eval()

        test_results = {'mse':[], 'psnr': [], 'ssim': [], 'lpips': []}
        start_time = time.time()
        for i in range(test_steps):
            x = next(testing_batch_generator)
            results = eval_step(x, models, opt, device, lpips_model)
            for key in test_results.keys():
                test_results[key] += results[key].cpu().detach().numpy().tolist()
        end_time = time.time()
        mse_value = np.mean(np.asarray(test_results['mse']))
        psnr_value = np.mean(np.asarray(test_results['psnr']))
        ssim_value = np.mean(np.asarray(test_results['ssim']))
        lpips_value = np.mean(np.asarray(test_results['lpips']))
        print('Epoch [%03d]: val mse: %.5f | val psnr: %.5f | val ssim: %.5f | val lpips: %.5f | testing time: %.5f' %
              (epoch, mse_value, psnr_value, ssim_value, lpips_value, end_time-start_time))

        all_metrics['psnr'].append(psnr_value)
        all_metrics['ssim'].append(ssim_value)
        all_metrics['lpips'].append(lpips_value)

        # --------- Save best model ------------------------------------
        to_save = models
        to_save['opt'] = opt
        to_save['epoch'] = epoch
        if psnr_value > best_psnr_value:
            print('-----> Best psnr model, psnr=', psnr_value)
            torch.save(to_save,'%s/best_psnr_model.pth' % (save_path))
            best_psnr_value = psnr_value
        if ssim_value > best_ssim_value:
            print('-----> Best ssim model, ssim=', ssim_value)
            torch.save(to_save, '%s/best_ssim_model.pth' % (save_path))
            best_ssim_value = ssim_value

        # --------- Save model at every epoch------------------------------------
        checkpoints_path = os.path.join(save_path, 'checkpoints')
        jsonfile = os.path.join(save_path, 'checkpoints', 'log.json')
        f = open(jsonfile, "w")
        f.write(json.dumps(all_metrics))
        f.close()
        torch.save(to_save, '%s/epoch_%03d_model.pth' % (checkpoints_path, epoch))
        if os.path.exists('%s/epoch_%03d_model.pth' % (checkpoints_path, epoch - 1)):
            os.remove('%s/epoch_%03d_model.pth' % (checkpoints_path, epoch - 1))

def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)  # Choose GPU for training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD

    dataset_root = opt.dataset.upper()
    # Create the folder to save model and checkpoint
    save_path = os.path.join(dataset_root, 'save_models_v2')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, "checkpoints")):
        os.mkdir(os.path.join(save_path, "checkpoints"))

    # Read dataset
    train_dataset = get_data(os.path.join(dataset_root, 'train.csv'))
    test_dataset = get_data(os.path.join(dataset_root, 'test.csv'))
    print('Train set:', len(train_dataset))
    print('Test set:', len(test_dataset))

    train_dataset = clean_data(train_dataset, opt.n_past + opt.n_future, MAX_FRAMES=3000)
    test_dataset = clean_data(test_dataset, opt.n_past + opt.n_eval, MAX_FRAMES=3000)
    print('Train set after clean:', len(train_dataset))
    print('Test set after clean:', len(test_dataset))

    training_steps = opt.training_steps
    if len(train_dataset) > training_steps * opt.batch_size:
        training_steps = len(train_dataset) * 2 //opt.batch_size

    test_steps = 0
    for row in test_dataset:
        test_steps += int(row[3]) // (opt.n_past + opt.n_eval)
    if (opt.dataset.upper() == 'KTH') or (opt.dataset.upper() == 'KITTI'):
        test_steps = 400

    training_steps = 10
    test_steps = opt.batch_size + 1
    print('Training steps:', training_steps)
    print('Test steps:', test_steps//opt.batch_size)

    training(opt, dataset_root=dataset_root, train_dataset=train_dataset, test_dataset=test_dataset,
             save_path=save_path, training_steps=training_steps, test_steps=test_steps//opt.batch_size, device=device)

if __name__ == '__main__':
    opt = get_parser()
    main(opt)