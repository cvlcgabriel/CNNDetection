import os
import csv
import torch
import numpy as np
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
from sklearn.metrics import confusion_matrix
 
# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]
 
print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default
 
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()
 
    # Retirando valores para matriz de confusão
    acc, ap, _, _, y_true, y_pred = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))
 
    # Gerando matriz de confusão
    y_pred_binary = (y_pred > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    print("Confusion Matrix for {}: \n{}".format(val, conf_matrix))
 
# Salvando o resultado no arquivo CSV
csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
 