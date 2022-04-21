import os, torch, config, datetime
from evaluation.evaluate_hpatches import HPatchesEvaluator
from evaluation.extract_hpatches import MultiScaleFeatureExtractor


class HPatchesExtractAndEvaluate:
    def __init__(self, args, exp_name, split='debug'):
        self.path = args.hpatches_path

        self.extractor = MultiScaleFeatureExtractor(args, exp_name, split)
        self.evaluator1 = HPatchesEvaluator(self.path, split, 1000, False)
        self.evaluator2 = HPatchesEvaluator(self.path, split, 1000, True) 

    def run(self, model1=None):
        with torch.no_grad():
            self.extractor.extract_hpatches(model1)
            print("\n================ without outlier rejection =======================")
            ret1 = self.evaluator1.compute(os.path.join(self.extractor.get_save_feat_dir(), self.path)) 
            print("\n================ with outlier rejection =======================")
            ret2 = self.evaluator2.compute(os.path.join(self.extractor.get_save_feat_dir(), self.path))  


if __name__ == "__main__":
    args = config.get_config()

    args.exp_name = 'eval_with_extract' + datetime.datetime.now().__format__('_%m%d_%H%M%S')
    args.eval_split = 'full'
    
    with torch.no_grad():
        hpatches_val = HPatchesExtractAndEvaluate(args, exp_name=args.exp_name, split=args.eval_split)
        hpatches_val.run()
        
