import xml.etree.ElementTree as ET
from pathlib import Path

class CfgRead:

    def __init__(self, xml_file, output_path='.'):
        """
            PhysiCell config file reading and writing class
        """
        # dictionary of user parameters and their values from xml
        self.data = self._read_xml(xml_file, output_path)
        # filename for internal use
        self._file = Path(output_path) / xml_file


    def _read_xml(self, xml_file, output_path='.'):
        """
        Reads xml file and saves all of the user parameters to a dictionary
        """

        output_path = Path(output_path)
        xml_file = output_path / xml_file
        tree = ET.parse(xml_file)

        print('Reading {}'.format(xml_file))

        root = tree.getroot()

        # Get childs of user_parameters node and their values
        user_params = root.find("user_parameters")
        UserParameters = {}
        for child in user_params:
            UserParameters[child.tag] = child.text

        return UserParameters

    def write_new_param(self, parameter="switching_time", value = '10'):
        """
        Write new switching_time to the file

        Parameters
        ----------
        parameter: str
                name of the parameter to be rewritten

        value: str
                value of the parameter to be written to the xml file

        """

        xml_file = self._file
        tree = ET.parse(xml_file)

        root = tree.getroot()
        user_params = root.find("user_parameters")

        # Find child node of interest and rewrite it
        node = user_params.find(parameter)
        node_value = node.text
        print("node value before = ",node_value)
        node.text = value
        print("new node value = ", node.text)

        tree.write(xml_file)
        return


