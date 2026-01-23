import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=4096, type=int, help='batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=400, type=int, help='number of epochs')
	parser.add_argument('--latdim', default=32, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--data', default='lastfm', type=str, help='name of dataset')
	parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for contrative learning')
	parser.add_argument("--ib_reg", type=float, default=0.1, help='weight for information bottleneck')
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default=-1, type=int, help='indicates which gpu to use')
	parser.add_argument('--lambda0', type=float, default=1e-4, help='weight for L0 loss on laplacian matrix.')
	parser.add_argument('--gamma', type=float, default=-0.45)
	parser.add_argument('--zeta', type=float, default=1.05)
	parser.add_argument('--init_temperature', type=float, default=2.0)
	parser.add_argument('--temperature_decay', type=float, default=0.98)
	parser.add_argument("--eps", type=float, default=1e-3)
	parser.add_argument("--seed", type=int, default=421, help="random seed")
	parser.add_argument("--exp", type=str, default="AdaGCL_test")
	parser.add_argument("--debug", action='store_true')
	parser.add_argument('--hints', nargs='*', help='hits of testdata recall.', default=[10, 20, 40])
    # keep evaluation standard consistent: Recall denominator = |GT|
	# checkpointing
	parser.add_argument('--save_best', type=int, default=1, help='1 to save best checkpoint during training')
	parser.add_argument('--resume_ckpt', type=str, default='', help='path to checkpoint for resuming training')
	# deprecated: hard negative sampler not used in current pipeline
	parser.add_argument('--acl_ratio', type=float, default=1.0)
	parser.add_argument('--acl_mlp_nums', type=int, default=1)
	parser.add_argument('--attention_type', type=str, default='original')
	parser.add_argument('--cl_type', type=str, default='gcl', help= "gcl | acl | mix")

	## diffusion relative
	parser.add_argument("--use_diff_gcl", type=int, default=0)
	parser.add_argument('--noise_scale', type=float, default=0.1)
	parser.add_argument('--noise_min', type=float, default=0.0001)
	parser.add_argument('--noise_max', type=float, default=0.02)
	parser.add_argument('--diff_steps', type=int, default=5)
	parser.add_argument('--d_emb_size', type=int, default=10)
	parser.add_argument('--denoise_dims', type=str, default='[64]')
	# deprecated: not used in current RaDAR diffusion
	parser.add_argument('--diff_alpha', type=float, default=0.3)
	parser.add_argument('--norm', type=bool, default=True)
	# deprecated: unused placeholder

	parser.add_argument('--lambda_diff', type=float, default=0.1)

	# diffusion regularizer
	parser.add_argument('--lambda_ddr', type=float, default=0.1, help='weight for DDR-style denoising regularizer')
	parser.add_argument('--ddr_warmup', type=int, default=20, help='epochs to warm up before applying DDR loss')
	parser.add_argument('--use_weighted_edges', type=int, default=0, help='1 to treat input matrices as weighted edges')


	# allow external scripts to pass extra args without failing
	args, _ = parser.parse_known_args()
	return args


args = ParseArgs()
