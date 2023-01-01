import argparse

import feature_extractor
import rank

def main(args):
    query_path, reference_path = feature_extractor.main(args)
    rank.main(args, query_path, reference_path)

if __name__ == '__main__':
    #feature extractor args 
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--root_path', type=str, default='/home/image-retrieval/ndir_simulated/')
    parser.add_argument('--data_path', type=str, default='/home/image-retrieval/ndir_simulated/dataset')
    parser.add_argument('--feature_path', type=str, default='/home/image-retrieval/ndir_simulated/features')
    parser.add_argument('--result_path', type=str, default='/home/image-retrieval/ndir_simulated/result')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='resnet50')  
    parser.add_argument('--checkpoint', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')    
    
    # #intermediate layer
    # parser.add_argument('--target_layer', type=int, default=3)

    # rank args
    parser.add_argument('-k', '--topk', type=int, default=None)

    args = parser.parse_args()

    main(args)