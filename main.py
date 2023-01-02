import argparse

import feature_extractor
import rank

def main(args):
    #import pdb; pdb.set_trace()
    feat_path_list = feature_extractor.main(args)
    print(f"feat_path_list is {feat_path_list}")
    for (query_path, reference_path) in feat_path_list:
        print(f"===="*30)
        print(f"feat_path is {query_path, reference_path}\n\n")
        rank.main(args, query_path, reference_path)

if __name__ == '__main__':
    # feature extractor args 
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/home/image-retrieval/ndir_simulated/')
    parser.add_argument('--data_path', type=str, default='/home/image-retrieval/ndir_simulated/dataset', help='signboard images directory')
    parser.add_argument('--feature_path', type=str, default='/home/image-retrieval/ndir_simulated/features')
    parser.add_argument('--result_path', type=str, default='/home/image-retrieval/ndir_simulated/result')
    parser.add_argument('--image_mode', type=str, default='original', help='original | keep_latio')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--model_mode', type=str, default='ATTN', help='CNN | ATTN')  
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--checkpoint', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')    

    # rank args
    parser.add_argument('-k', '--top_k', type=int, default=10)
    args = parser.parse_args()

    main(args)