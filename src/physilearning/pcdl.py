import zmq
import re
import pandas as pd
import os

class PhysiCellDataListener:
    """
    Listens for data from a PhysiCell simulation.
    """

    def __init__(
        self,
        port=5556,
    ):

        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.socket.connect(f'ipc:///tmp/0_cell_data')
        self.message = None

    def get_data(self):
        """
        Returns the data from the simulation.
        """
        message = self.socket.recv().decode('utf-8')
        self.message = message
        return message

    def _message_to_df(self, message):
        """
        Converts the message to a dataframe.
        """
        df = pd.DataFrame()
        # find all the words in the message
        word_pattern = r'\w+:'
        words = re.findall(word_pattern, message)

        # loop through the words and find the parameters
        for word in words:
            parameter = word[:-1]
            next_parameter = words[words.index(word)+1][:-1]
            if next_parameter == 'end':
                df[parameter] = self._find_parameter(
                    parameter=parameter, next_parameter=next_parameter, type='float', message=message)
                break
            else:
                df[parameter] = self._find_parameter(
                    parameter=parameter, next_parameter=next_parameter, type='float', message=message)

        return df

    def _find_parameter(self, parameter: str = 'ID', next_parameter: str = 'x',
                        type: str = 'int', message: str = None):
        """
        Finds a parameter in the message.
        """
        start_idx = message.find(f'{parameter}:')+len(f'{parameter}:')
        end_idx = message.find(f',;{next_parameter}:')
        truncated_message = message[start_idx:end_idx]
        list_of_str_params = truncated_message.split(',')
        integer_params = ['ID', 'barcode', 'type']
        if parameter in integer_params:
            return [int(x) for x in list_of_str_params]
        else:
            return [float(x) for x in list_of_str_params]


    def write_to_hdf(self, message: str, path: str):
        """
        Writes the dataframe to hdf file.
        """

        time = re.findall(r'\d+', message)[0]
        df = self._message_to_df(message)
        df.to_hdf(path, f'time_{time}')

        return


if __name__ == '__main__':
    os.chdir('/home/saif/Projects/PhysiLearning')
    listener = PhysiCellDataListener()
    while True:
        message = listener.get_data()
        listener.write_to_hdf(message=message, path='test_data.h5')
        print(listener._message_to_df(message))
