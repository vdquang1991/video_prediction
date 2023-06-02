import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Training Video Prediction')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--dataset', default='kitti', help='dataset to train with')
    parser.add_argument('--n_past', type=int, default=10, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
    parser.add_argument('--n_eval', type=int, default=25, help='number of frames to predict during eval')
    parser.add_argument('--n_channel', default=3, type=int)
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--skip_type', default='residual', type=str, help='residual or concat')
    parser.add_argument('--act', choices=['relu', 'leaky', 'elu', 'swish'], default='swish', help='activation type')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate init')
    parser.add_argument('--beta_gdl', default=1., type=float, help='loss weight for gdl loss')
    parser.add_argument('--beta_bce', default=1., type=float, help='loss weight for binary cross-entropy loss')
    parser.add_argument('--beta_kld', default=1., type=float, help='loss weight for kld loss')
    parser.add_argument('--c', default=0.2, type=float, help='c factor for resampling image')
    parser.add_argument('--init_filters', type=int, default=64, help='init number of filters')
    parser.add_argument('--g_dim', type=int, default=128, help='Output of encoder and input of decoder')
    parser.add_argument('--z_dim', type=int, default=50, help='dimension of z_t')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer in LSTM Cell')
    parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers for prior')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers for posterior')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers for predictor')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--sch_sampling', type=int, default=0,
                        help='if given an integer, scheduled sampling will be used. inverse sigmoid with k.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--training_steps', type=int, default=2000, help='number of total iter for each epoch')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--no_test', action='store_false', dest='test',
                        help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    # parser.add_argument('--model_lpips', choices=['net-lin', 'net'], default='net-lin', help='net-lin or net')
    # parser.add_argument('--net', choices=['squeeze', 'alex', 'vgg'], default='vgg', help='squeeze, alex, or vgg')
    # parser.add_argument('--version', type=str, default='0.1')
    args = parser.parse_args()
    args.sc_prob = 1
    return args

