from physilearning.tools.xml_reader import CfgRead
import pytest

@pytest.mark.skip(reason="Not implemented")
def test_XML_read():

    xml_reader = CfgRead('../src/PhysiCell_V_1.10.4_src/config/PhysiCell_settings.xml')

    assert xml_reader.data

@pytest.mark.skip(reason="Not implemented")
def test_XML_write():

    xml_reader = CfgRead('../src/PhysiCell_V_1.10.4_src/config/PhysiCell_settings.xml')
    xml_reader.write_new_param(parent_nodes=['save', 'full_data'], parameter="enable", value='true')

    assert True