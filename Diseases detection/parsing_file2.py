import argparse
import json
def create_parser_disease_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default = 2)
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--es', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--file2read', type=int, default = 2)
    #parser.add_argument('--WANTED_TESTS1', type=str, default = '[Spontaneous 1 pursuit 1 Saccade 1 Gazetest 1 optokinetic 1]')
    #parser.add_argument('--WANTED_TESTS', type=json.loads, default = '{"Spontaneous nystagmus":1,"pursuit":1,"Saccade":1,"Gaze test":1,"optokinetic":1}')    
    #parser.add_argument('--WANTED_FEATURES', type=json.loads, default = '{"c0_left":1,"c1_left":1,"area":1,"axis_major_left":0,"axis_minor_left":0,"c0_right":0,"c1_right":0,"area.1":0, "axis_major_right":0, "axis_minor_right":0}')    

    parser.add_argument('--WANTED_TESTS', type=str, default = 'N')    
    parser.add_argument('--WANTED_FEATURES', type=str, default = 'X')    

    #parser.add_argument('--WANTED_FEATURES', type=str, default = '[c0_left,c1_left,c0_right,c1_right,area,area.1]')
    parser.add_argument('--TS', type=int, default = 500)
    parser.add_argument('--SIZE', type=int, default = 50)
    #parser.add_argument('--WANTED_ALGS', type=json.loads,default = '{"GASF":0,"GADF":0,"MTF":0,"RP":0,"GASF_MTF":0}')
    parser.add_argument('--WANTED_ALGS', type=str,default = 'GASF')
    parser.add_argument('--USED_MODEL_ARCH', type=str,default = 'CNN')

    #parser.add_argument('--model', type=str, default = 'model1')
    parser.add_argument('--output', type=str, default = 'output_file')
    return parser

    