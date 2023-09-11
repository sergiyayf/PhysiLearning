import zmq
import re
import pandas as pd

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
        id_list = self._find_parameter(parameter='ID', next_parameter='x', type='int', message=message)
        x_pos = self._find_parameter(parameter='x', next_parameter='y', type='float', message=message)
        y_pos = self._find_parameter(parameter='y', next_parameter='z', type='float', message=message)
        z_pos = self._find_parameter(parameter='z', next_parameter='barcode', type='float', message=message)
        barcode = self._find_parameter(parameter='barcode', next_parameter='type', type='int', message=message)
        type = self._find_parameter(parameter='type', next_parameter='elapsed_time_in_phase', type='int', message=message)
        elapsed_time_in_phase = self._find_parameter(parameter='elapsed_time_in_phase', next_parameter='end', type='float', message=message)
        df['ID'] = id_list
        df['x'] = x_pos
        df['y'] = y_pos
        df['z'] = z_pos
        df['barcode'] = barcode
        df['type'] = type
        df['elapsed_time_in_phase'] = elapsed_time_in_phase

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

        if type == 'int':
            return [int(x) for x in list_of_str_params]
        elif type == 'float':
            return [float(x) for x in list_of_str_params]
        else:
            raise ValueError(f'Invalid type: {type}')



if __name__ == '__main__':
    listener = PhysiCellDataListener()
    while True:
        message = listener.get_data()
        df = listener._message_to_df(message)
        print(df)