import argparse
import sys


def get_argparser():
    """Add arguments for parser.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    # i/o
    parser.add_argument('--dataset_name',
                        type=str,
                        default='simulation',
                        help='exp type ["simulation" or "ihdp" or "news"]')
    
    parser.add_argument('--data_dir',
                        type=str,
                        default='dataset/simu',
                        help='dir of eval dataset')

    # data
    parser.add_argument('--num_obs', type=int, default=2000, help='The number of data points used for evaluation')
    parser.add_argument('--t_noise_dist', type=str, default='normal', help='type of the noise for t')
    parser.add_argument('--me_dist', type=str, default='normal', help='distribution of measurement error')
    parser.add_argument('--me_std', type=float, default=0.2, help='std of the measurement error')

    # training
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')

    # model
    parser.add_argument('--model', type=str, default='vf_vcnet', help='model name')
    parser.add_argument('--y_coeff', type=float, default=1, help='coefficient for outcome error loss')
    parser.add_argument('--alpha', type=float, default=1.5, help='alpha for measurement error loss')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for kl loss')
    parser.add_argument('--n_components', type=int, default=1, help='number of components for mixture of gaussians')

    # exp_setting
    parser.add_argument('--seed', type=int, default=1, help="Random seed.")
    parser.add_argument('--exp_id', type=str, default='reg_test', help='exp id')
    parser.add_argument('--reg', type=bool, default=True, help='whether to use target regression')

    # plot adrf
    parser.add_argument('--plt_adrf', type=bool, default=True, help='whether to plot adrf curves. (only run two methods if set true; '
                                                                    'the label of fig is only for drnet and vcnet in a certain order)')

    args, unknown = parser.parse_known_args()
    return args
