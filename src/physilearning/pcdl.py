import zmq
import re
import pandas as pd
import click
import yaml


class PhysiCellDataListener:
    """
    Listens for data from a PhysiCell simulation.

    :param port: The port to listen on.
    :param jobid: The jobid of the simulation.
    """

    def __init__(
        self,
        port=5556,
        jobid=0,
    ):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.transport_address = config['env']['PcEnv']['transport_address']
        self.port = port
        self.jobid = jobid
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.socket.connect(f'ipc://{self.transport_address}{self.jobid}{self.port}_cell_data')
        self.message = None
        self.run = 0

    def get_data(self):
        """
        Returns the data from the simulation.
        """
        message = self.socket.recv().decode('utf-8')
        self.message = message
        return message

    def message_to_df(self, message):
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
                    parameter=parameter, next_parameter=next_parameter, message=message)
                break
            else:
                df[parameter] = self._find_parameter(
                    parameter=parameter, next_parameter=next_parameter, message=message)

        return df

    def _find_parameter(self, parameter: str = 'ID', next_parameter: str = 'x', message: str = None):
        """
        Finds a parameter in the message.
        """
        start_idx = message.find(f'{parameter}:')+len(f'{parameter}:')
        end_idx = message.find(f',;{next_parameter}:')
        truncated_message = message[start_idx:end_idx]
        list_of_str_params = truncated_message.split(',')
        integer_params = ['ID', 'barcode', 'type']
        string_params = ['sequence']
        if parameter in integer_params:
            return [int(x) for x in list_of_str_params]
        elif parameter in string_params:
            return [x for x in list_of_str_params]
        else:
            return [float(x) for x in list_of_str_params]

    def write_to_hdf(self, message: str, path: str):
        """
        Writes the dataframe to hdf file.
        """
        time = re.findall(r'\d+', message)[0]
        df = self.message_to_df(message)
        #if str(time) == '0':
        self.run += 1
        df.to_hdf(path, f'run_{self.run}/time_{time}')
        return


@click.command()
@click.option('--jobid', default=0, help='ID of the job')
@click.option('--port', default=0, help='ID of the task')
def main(jobid, port):

    listener = PhysiCellDataListener(jobid=jobid, port=port)
    print("Listener initiated")
    while True:
        message = listener.get_data()
        listener.write_to_hdf(message=message, path=f'pcdl_data_job_{jobid}_port_{port}.h5')
        print(listener.message_to_df(message))


if __name__ == '__main__':
    main()
