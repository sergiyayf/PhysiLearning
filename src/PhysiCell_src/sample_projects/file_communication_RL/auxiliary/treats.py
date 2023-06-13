import scipy.io as sio
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

class Treatment():

    def __init__(self):
        self.filename = "Treatment_schedule.mat"
        self.dir = './../PhysiCell_V_1.10.4'
        self.schedule = None

    def get_current_treatment(self):
        """ Get current treatment schedule """
        mat_file = Path(self.dir)/self.filename
        A = sio.loadmat(mat_file)['schedule']
        return A

    def set_treatment_file_for_current_sim(self, config = r'.\..\PhysiCell_V_1.10.4\config', settings_file = 'PhysiCell_settings.xml', treatment_interval = 60, mode = "default"):
        """ Create a matlab file for treatment decisions """
        # Read config file
        output_path = Path(config)
        xml_file = output_path / settings_file
        tree = ET.parse(xml_file)

        print('Reading {}'.format(xml_file))

        root = tree.getroot()

        # Get current simulating time
        overall_node = root.find('overall')
        max_time = int(overall_node.find('max_time').text)

        if mode == "default":
            " default means no treatment "
            # Create a numpy array of length simulation_time/treatment_interval, first column -> time, second -> treatment, third -> if decision is made
            schedule = np.zeros((int(max_time/treatment_interval)+1,3))

            # Write treatment timing
            schedule[:,0] = np.arange(int(max_time/treatment_interval)+1)*treatment_interval
            # Write for zero time that decision has been made
            schedule[0,2] = 1
            self.schedule = schedule
            # Save numpy array to treatment matlab file
            mdict = {'schedule': schedule}
            mat_file = Path(self.dir)/self.filename
            sio.savemat(mat_file, mdict, format = '4')

        return

    def change_treatment(self, time, decision):
        """ Change a specific treatment decision """

        # Read treatment schedule to an array
        schedule = self.get_current_treatment()

        # Change the specified decision
        schedule[time, 1] = decision
        # Write that decision has been made
        schedule[time, 2]  = 1
        # Save back to matlab
        mdict = {'schedule': schedule}
        mat_file = Path(self.dir) / self.filename
        sio.savemat(mat_file, mdict, format='4')
        return

if __name__ == '__main__':

    t = Treatment()
    t.set_treatment_file_for_current_sim()
    # B = t.get_current_treatment()
    # print(B)
    t.change_treatment(np.arange(14,25,1),np.zeros_like(np.arange(14,25,1)))
    B = t.get_current_treatment()
    print(B)
