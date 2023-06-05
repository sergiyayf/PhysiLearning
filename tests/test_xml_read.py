from physilearning.tools.xml_reader import CfgRead
import pytest


def test_XML_read():

    xml_reader = CfgRead('./src/PhysiCell_V_1.10.4_src/config/PhysiCell_settings.xml')

    assert xml_reader.data


def test_XML_read_value():

    xml_reader = CfgRead('./src/PhysiCell_V_1.10.4_src/config/PhysiCell_settings.xml')
    value = xml_reader.read_value(parent_nodes=['domain'], parameter="dx")

    assert value == '20'


def test_XML_write():

    xml_reader = CfgRead('./src/PhysiCell_V_1.10.4_src/config/PhysiCell_settings.xml')
    real_value = xml_reader.read_value(parent_nodes=['save', 'full_data'], parameter="enable")
    xml_reader.write_new_param(parent_nodes=['save', 'full_data'], parameter="enable", value='something else')
    written_value = xml_reader.read_value(parent_nodes=['save', 'full_data'], parameter="enable")
    assert real_value != written_value
    xml_reader.write_new_param(parent_nodes=['save', 'full_data'], parameter="enable", value=real_value)
    written_value = xml_reader.read_value(parent_nodes=['save', 'full_data'], parameter="enable")
    assert real_value == written_value


def test_XML_write_new_custom_param():

    xml_reader = CfgRead('./src/PhysiCell_V_1.10.4_src/config/PhysiCell_settings.xml')
    real_value = xml_reader.read_value(parent_nodes=['user_parameters'], parameter="treatment")
    xml_reader.write_new_custom_param(parameter="treatment", value='something else')
    written_value = xml_reader.read_value(parent_nodes=['user_parameters'], parameter="treatment")
    assert real_value != written_value
    xml_reader.write_new_custom_param(parameter="treatment", value=real_value)
    written_value = xml_reader.read_value(parent_nodes=['user_parameters'], parameter="treatment")
    assert real_value == written_value
