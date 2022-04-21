import argparse
from evaluation.evaluate_hpatches import HPatchesEvaluator

if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument("--hpatches_path", required=False, default='./datasets/hpatches-sequences-release', type=str, help='location of hpatches directory.')
    config.add_argument("--nfeatures", required=False, default=1000, type=int)
    config.add_argument('--load_data', required=False, default='', type=str)
    config.add_argument('--eval_split', required=False, default='full', type=str)
    config.add_argument('--outlier_rejection', required=False, default=False, type=bool)
    config.add_argument('--outlier_threshold', required=False, default=30, type=int)

    args = config.parse_args()

    eval = HPatchesEvaluator(args.hpatches_path, split=args.eval_split, nfeatures=args.nfeatures, \
        outlier_rejection=args.outlier_rejection, outlier_threshold=args.outlier_threshold)
    eval.compute(load_data)
