import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, required=True, default = 1)
    # parser.add_argument('--lr', type=float, required=True, default = 0.0001)
    # parser.add_argument('--es', type=int, required=True, default = 1)
    # parser.add_argument('--wanted', type=str, required=True, default = 'pis')
    # parser.add_argument('--file2read', type=int, required=True, default = 100)
    # parser.add_argument('--backbon', type=str, required=True, default = 'mine')
    # parser.add_argument('--weights',  type=str, default='[0.95,0.1,0.1,0.1]')
    # parser.add_argument('--output', type=str, required=True, default = 'Test_output')

    # parser.add_argument('--batch_size', type=int, required=True, default = 128)
    # parser.add_argument('--DS_NAME', type=str, required=True, default = 's-openeds')
    # parser.add_argument('--metric_thr', type=float, required=True, default = 0.5)
    # parser.add_argument('--model_weights', type=str, required=True, default = '')
    parser.add_argument('--epochs', type=int, default = 1)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--es', type=int, default = 1)
    parser.add_argument('--wanted', type=str, default = 'p')
    parser.add_argument('--file2read', type=int, default = 100)
    parser.add_argument('--backbon', type=str, default = 'mine')
    parser.add_argument('--weights',  type=str, default='[1,0,0,0]')
    parser.add_argument('--output', type=str, default = 'Test_output')

    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--DS_NAME', type=str, default = 'NN_human_mouse_eyes')
    # parser.add_argument('--DS_NAME', type=str, default = 'MOBIUS')
    # parser.add_argument('--DS_NAME', type=str, default = 's-openeds')
    
    parser.add_argument('--metric_thr', type=float, default = 0.5)
    parser.add_argument('--model_weights', type=str, default = '')
    
    return parser




def create_parser_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_PATH', type=str, default = '')    
    parser.add_argument('--DS_NAME', type=str, default = '')
    parser.add_argument('--IMG_PATH', nargs='+', default = '')    
    parser.add_argument('--VIDEO_PATH', type=str, default = '')    
    parser.add_argument('--OUTPUT_PATH', type=str, default = '')    
    parser.add_argument('--SEGMENT', type=int, required=False) 
    parser.add_argument('--imgExt', type=str, required=False) 
    parser.add_argument('--maskExt', type=str) 
    parser.add_argument('--WANTED', type=str, default = 'p') 
    parser.add_argument('--CLASSES_NUM', type=int, default = 2) 
    parser.add_argument('--CLANE', type=int, default = 1) 
    
       


    return parser


def create_parser_eval_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_PATH', type=str, default = '')    
    parser.add_argument('--WANTED', type=str, default = 'p') 
    parser.add_argument('--CLASSES_NUM', type=int, default = 2) 
    return parser

    