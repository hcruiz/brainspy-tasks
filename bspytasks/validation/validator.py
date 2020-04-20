import os
import numpy as np
import matplotlib.pyplot as plt
# from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspyproc.bspyproc import get_processor
from bspyproc.utils.control import get_control_voltage_indices
from bspyproc.utils.waveform import generate_slopped_plato
# TODO: put these params in processors
MAX_CLIPPING_VALUE = np.array([1.5])
MIN_CLIPPING_VALUE = np.array([-1.5])


class Hardware_Validator:

    def __init__(self, configs):
        self.configs = configs

        self.validation_dir = configs['data_dir']
        self.show_plots = configs['show_plots']
        self.input_indices = configs['processor']["input_indices"]
        self.nr_electrodes = configs['processor']["input_electrode_no"]
        self.cv_indices = get_control_voltage_indices(self.input_indices, self.nr_electrodes)
        self.slope_length = configs["processor"]["waveform"]["slope_lengths"]
        self.amplitude_lengths = configs["processor"]["waveform"]["amplitude_lengths"]
        assert self.slope_length > 0, f"Slopes cannot be zero! slope_length=={self.slope_length}"
        assert self.amplitude_lengths > 10, f"Input plateaus cannot be zero! amplitude_lengths=={self.amplitude_lengths}"

    def clip(self, x, max_value, min_value):
        x[x > max_value] = max_value
        x[x < min_value] = min_value
        return x

    def generate_sloped_values(self, values, total_length):
        return values[np.newaxis] * generate_slopped_plato(self.slope_length, total_length)[:, np.newaxis]

    def generate_input_matrix(self, inputs, control_voltages):
        inp_matrix = np.empty((inputs.shape[0], self.nr_electrodes))
        inp_matrix[:, self.input_indices] = inputs
        inp_matrix[:, self.cv_indices] = self.generate_sloped_values(control_voltages, inputs.shape[0])
        return inp_matrix

    def validate_prediction(self, name, inputs, control_voltages, predictions, mask):
        print('==========================================================================================')
        print(f"Validating model prediction of {name}...")

        inputs = self.clip(inputs, max_value=MAX_CLIPPING_VALUE, min_value=MIN_CLIPPING_VALUE)
        control_voltages = self.clip(control_voltages, max_value=MAX_CLIPPING_VALUE, min_value=MIN_CLIPPING_VALUE)
        input_matrix = self.generate_input_matrix(inputs, control_voltages)
        # assert len(input_matrix) == len(predictions), f"Prediction {len(input_matrix)} and input matrix {len(predictions)} have not the same lenghts"
        self.configs["processor"]["shape"] = len(input_matrix)
        # if self.configs["processor"]["shape"] != len(predictions):
        #     print(f"Changing shape key of processor from value {self.configs['processor']['shape']} to {len(predictions)}")
        #     self.configs["processor"]["shape"] = len(predictions)
        assert not all(mask), "Mask indicates there is no ramping. Stop measuring..."
        processor = get_processor(self.configs['processor'])
        measurement = processor.get_output(input_matrix)
        try:
            processor.close_tasks()
        except AttributeError:
            print("WARNING: close_task method call ignored")
        # TODO: to estimate the MSE save predicted_dnpuoutput when training
        error = np.nan
        var_measurement = np.nan
        # error = ((measurement[mask, 0] - predicted_dnpuoutput[mask]) ** 2).mean()  # TODO: predictions should be 2 dim array with last dim singleton
        # print(f'MSE: {str(error)}')
        # var_measurement = np.var(measurement[mask], ddof=1)
        # print(f'(var) NMSE: {100*error/var_measurement} %')
        self.plot_validation(name, measurement[mask], predictions, show_plots=self.show_plots, save_dir=self.validation_dir)
        print('==========================================================================================')
        return error, var_measurement, measurement[mask]

    def plot_validation(self, name, output, prediction, show_plots, save_dir=None):
        fig, ax1 = plt.subplots()
        plt.title(name)
        color = 'tab:red'
        ax1.set_xlabel('time steps')
        ax1.set_ylabel('Device output (nA)', color=color)
        ax1.plot(output, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('cl. 1 probability', color=color)  # we already handled the x-label with ax1
        ax2.plot(prediction, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, name + "_output"))
        if show_plots:
            plt.show()
        plt.close()