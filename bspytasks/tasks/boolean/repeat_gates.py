'''This function repeats the search over gates that were not found.
It assumes there is a file output of the gate validator (validate_binary); or any file as dictionary where missed gates are saved.
'''
import numpy as np
import os

import bspytasks.tasks.boolean.gate_finder as finder

validation_dir = 'tmp/TEST/validation/'
data_file = os.path.join(validation_dir, 'result_arrays.npz')
with np.load(data_file) as data:
    mask = data['mask']
    inputs = data['inputs']
    old_predictions = data['output_array'][:, mask]
    targets_array = data['targets_array'][:, mask, np.newaxis]
    gate_array = data['gate_array'].astype('int')
    old_accuracy = data['accuracy_array']

data_file = os.path.join(validation_dir, 'validated_data.npz')
with np.load(data_file) as data:
    found_array = data['found_array'].astype('bool')

not_found = gate_array[~found_array]
accuracy_not_found = old_accuracy[~found_array]
old_pred_not_found = old_predictions[~found_array]
targets_not_found = targets_array[~not_found]

print(f'Data not found: \n {zip(not_found,accuracy_not_found)} \n {"===="*len(gate_array[0])}')

number_gates = len(not_found)
vc_dimension = len(gate_array[0])
gate_ = np.zeros((number_gates, vc_dimension))
accuracy_ = np.zeros(number_gates)
performance_ = np.zeros_like(accuracy_)
found_ = np.zeros_like(accuracy_)
correlation_ = np.zeros_like(accuracy_)
control_voltages_per_gate = np.zeros((number_gates, 5))  # TODO: un-hard code nr. dimensions
output_ = np.zeros((number_gates, len(old_predictions[1])))
targets_ = np.zeros((number_gates, len(inputs)))  # should be kept without ramps rather than the complete waveform

for nr, gate in enumerate(not_found):
    results = finder.find_single_gate('configs/benchmark_tests/capacity/template_gd_noVal.json', str(gate), len(gate))
    # Collect results needed in the validator
    # data['output_array']
    # data['mask']
    # data['targets_array'][:, mask, np.newaxis] ; targets should be kept without ramps rather than the complete waveform
    # data['inputs']
    # data['control_voltages_per_gate']
    # data['gate_array']
    gate_[nr] = results['gate']
    accuracy_[nr] = results['accuracy']
    performance_[nr] = results['best_performance']
    found_[nr] = results['found']
    correlation_[nr] = results['correlation']
    control_voltages_per_gate[nr] = results['control_voltages']
    targets_[nr] = results['encoded_gate'][:, 0]
    output_[nr] = results['best_output'][:, 0]

inputs = results['inputs'].detach().cpu().numpy()
mask = results['mask']
capacity = 1 - np.sum(~found_.astype('bool')) / len(found_array)
print(f'New predicted capacity: {capacity}')
os.mkdir(os.path.join(validation_dir, 'repeated_search'))
numpy_file = os.path.join(validation_dir, 'repeated_search', 'result_arrays')
np.savez(numpy_file,
         gate_array=gate_,
         accuracy_array=accuracy_,
         performance_array=performance_,
         found_array=found_array,
         correlation_array=correlation_,
         control_voltages_per_gate=control_voltages_per_gate,
         targets_array=targets_,
         output_array=output_, inputs=inputs, mask=mask)
print('---------------------------------------------')
