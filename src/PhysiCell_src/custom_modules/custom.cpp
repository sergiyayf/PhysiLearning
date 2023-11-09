/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# See VERSION.txt or call get_PhysiCell_version() to get the current version  #
#     x.y.z. Call display_citations() to get detailed information on all cite-#
#     able software used in your PhysiCell application.                       #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version x.y.z) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 14(2): e1005991, 2018                   #
#     DOI: 10.1371/journal.pcbi.1005991                                       #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#     llelized diffusive transport solver for 3-D biological simulations,     #
#     Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730  #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2021, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "./custom.h"
#include <time.h>

void create_cell_types( void )
{
	// set the random seed
	if (parameters.ints("random_seed") == 99 )
    {
        SeedRandom( time(NULL) );
    }
    else
    {
        SeedRandom( parameters.ints("random_seed") );
    }
	
	/* 
	   Put any modifications to default cell definition here if you 
	   want to have "inherited" by other cell types. 
	   
	   This is a good place to set default functions. 
	*/ 
	
	initialize_default_cell_definition(); 
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	cell_defaults.functions.volume_update_function = standard_volume_update_function;
	cell_defaults.functions.update_velocity = standard_update_cell_velocity;

	cell_defaults.functions.update_migration_bias = NULL; 
	cell_defaults.functions.update_phenotype = update_cell_and_death_parameters_O2_based; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
 	cell_defaults.parameters.o2_proliferation_saturation = 30.0;
	cell_defaults.parameters.o2_proliferation_threshold = 8.0;
	cell_defaults.parameters.o2_necrosis_threshold = 2.0;
	cell_defaults.parameters.o2_necrosis_max = 0.0;
	cell_defaults.custom_data.add_variable( "treatment" , "dimensionless", 0.0 );
	/*
	   This parses the cell definitions in the XML config file. 
	*/
	
	initialize_cell_definitions_from_pugixml(); 

	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
	build_cell_definitions_maps(); 

	/*
	   This intializes cell signal and response dictionaries 
	*/

	setup_signal_behavior_dictionaries(); 	

	/*
       Cell rule definitions 
	*/

	setup_cell_rules(); 

	/* 
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	// cell_defaults.functions.update_phenotype = phenotype_function; 
	cell_defaults.functions.custom_cell_rule = custom_function; 
	cell_defaults.functions.contact_function = contact_function; 
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
	
	Cell_Definition* pSensitive = find_cell_definition("sensitive");

	pSensitive->functions.update_phenotype = susceptible_cell_phenotype_update_rule;
	display_cell_definitions( std::cout ); 
	
	return; 
}


/*
    Circular boundary conditions implementation
*/
void set_circular_boundary_conditions( void ) {

    std::vector<double> center_by_indices = {(double)microenvironment.mesh.x_coordinates.size()/2, (double)microenvironment.mesh.y_coordinates.size()/2, (double)microenvironment.mesh.z_coordinates.size()/2}; // find center voxel
    double system_radius = (double)microenvironment.mesh.x_coordinates.size()/2-1.0;
	// if there are more substrates, resize accordingly
	std::vector<double> bc_vector = {46.0, 40.0} ;
	for (unsigned int k = 0; k< microenvironment.mesh.z_coordinates.size(); k++) {
		// loop through y
		for (unsigned int j = 0; j< microenvironment.mesh.y_coordinates.size() ; j++) {
			// loop through x
			for (unsigned int I = 0; I< microenvironment.mesh.x_coordinates.size(); I++) {
				std::vector<double> current_index_position = {(double)I, (double)j, (double)k};

				if(dist(current_index_position,center_by_indices)>system_radius) {

					microenvironment.add_dirichlet_node( microenvironment.voxel_index(I,j,k), bc_vector);

				}
			}
		}
	}

	// initialize BioFVM
	get_default_microenvironment()->set_substrate_dirichlet_activation(0,true);
	get_default_microenvironment()->set_substrate_dirichlet_activation(1,false);

}

void activate_drug_dc(void)
{
	static int drug_index = microenvironment.find_density_index("drug");
	microenvironment.set_substrate_dirichlet_activation(drug_index,true);
	// change custom data treatment parameter in 0th cell
	(*all_cells)[0]->custom_data["treatment"] = 1.0;

	return;
}

void deactivate_drug_dc(void)
{
	static int drug_index = microenvironment.find_density_index("drug");
	microenvironment.set_substrate_dirichlet_activation(drug_index,false);
	(*all_cells)[0]->custom_data["treatment"] = 0.0;
	return; 
}

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
    // if chechlpointing is available load microenv from the mat file
    if (parameters.bools("enable_chkpt")){
        std::cout<<"Loading microenvironment from "<<parameters.strings("filename_chkpt")+"_microenvironment0.mat"<<std::endl;
        read_microenvironment_from_MultiCellDS_xml(microenvironment, parameters.strings("filename_chkpt"));
        set_circular_boundary_conditions();
    } else {
        set_circular_boundary_conditions();
    }

	return; 
}

void setup_round_tumoroid( void )
{
	// place a cluster of tumor cells at the center
    double cell_radius = cell_defaults.phenotype.geometry.radius; 
	double cell_spacing = 0.95 * 2.0 * cell_radius; 
	
	
		
	Cell* pCell = NULL;
	    
    int n = 0; 
    int resistant_cells = 1;
    int susceptible_cells = 4000;
    
    double tumor_radius = std::sqrt(resistant_cells+susceptible_cells)*cell_radius; // 250.0; 
    double x = 0.0; 
    double x_outer = tumor_radius; 
	double y = 0.0;
	double z = 0.0;
    double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1];
	double Zmin = microenvironment.mesh.bounding_box[2];
    double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4];
	double Zmax = microenvironment.mesh.bounding_box[5];
    double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin;
	double Zrange = Zmax - Zmin;
	if (Zmax>30){
		tumor_radius *=0.3 ;
	}
	while( n < resistant_cells)
	{  
        
        double r = tumor_radius +1; 
        while (r>0.2*tumor_radius) {
        x = Xmin + UniformRandom()*Xrange; 
        y = Ymin + UniformRandom()*Yrange;
        if( default_microenvironment_options.simulate_2D == false ){
        z = Zmin + UniformRandom()*Zrange;
        } else {
        z = 0.0;
        }
        r = norm( {x,y,z} );
        }
        
        pCell = create_cell( get_cell_definition("resistant") );         
		pCell->assign_position( {x,y,z} );
		n++; 
	} 
	while( n < resistant_cells+susceptible_cells)
	{  
        
        double r = tumor_radius +1; 
        while (r>tumor_radius || r<0.2*tumor_radius ) {
        x = Xmin + UniformRandom()*Xrange; 
        y = Ymin + UniformRandom()*Yrange;
        if( default_microenvironment_options.simulate_2D == false ){
        z = Zmin + UniformRandom()*Zrange;
        } else {
        z = 0.0;
        }
        r = norm( {x,y,z} );
        }
        
        pCell = create_cell( get_cell_definition("sensitive") ); 
        pCell->assign_position( {x,y,z} );
		n++; 
	}  

	return; 
}

void setup_tissue( void )
{
    if (parameters.bools("enable_chkpt")){
        std::cout<<"Loading cell positions from"<<parameters.strings("filename_chkpt")+"_cells.mat"<<std::endl;
        load_minimal_cells_physicell(parameters.strings("filename_chkpt"));
    } else {

	double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1]; 
	double Zmin = microenvironment.mesh.bounding_box[2]; 

	double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4]; 
	double Zmax = microenvironment.mesh.bounding_box[5]; 
	
	if( default_microenvironment_options.simulate_2D == true )
	{
		Zmin = 0.0; 
		Zmax = 0.0; 
	}
	
	double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin; 
	double Zrange = Zmax - Zmin; 
	
	// create some of each type of cell 
	
	Cell* pC;
	
	for( int k=0; k < cell_definitions_by_index.size() ; k++ )
	{
		Cell_Definition* pCD = cell_definitions_by_index[k]; 
		std::cout << "Placing cells of type " << pCD->name << " ... " << std::endl; 
		for( int n = 0 ; n < parameters.ints("number_of_cells") ; n++ )
		{
			std::vector<double> position = {0,0,0}; 
			position[0] = Xmin + UniformRandom()*Xrange; 
			position[1] = Ymin + UniformRandom()*Yrange; 
			position[2] = Zmin + UniformRandom()*Zrange; 
			
			pC = create_cell( *pCD ); 
			pC->assign_position( position );
		}
	}
	std::cout << std::endl; 
	
	// load cells from your CSV file (if enabled)
	load_cells_from_pugixml();

	// set the correct barcode for all of the initially created cells
	/*
	for (int i=0; i<(*all_cells).size(); i++){
        Cell* pCell = (*all_cells)[i];
        std::bitset<128> temp_bitset(pCell->ID);
        int left_most_bit = 0;

        for (int j = 0; j < 128; j++) {
            if (temp_bitset[j]) {
                left_most_bit = j;
            }
        }
        // save leftmost bit to custom data
        pCell->custom_data["left_most_bit"] = left_most_bit+6;
        for (int j = 0; j < 4; j++) {
            temp_bitset.set(left_most_bit+2+j);
        }
        pCell->barcode = temp_bitset;
    } */
	//setup_round_tumoroid();
	}
	return; 
}

std::vector<std::string> my_coloring_function( Cell* pCell )
{
	 static int cycle_start_index = live.find_phase_index( PhysiCell_constants::live );
	 static int cycle_end_index = live.find_phase_index( PhysiCell_constants::live );
	 double growth_rate = pCell->phenotype.cycle.data.transition_rate(cycle_start_index,cycle_end_index);
	 double max_growth_rate = pCell->parameters.pReference_live_phenotype->cycle.data.transition_rate(cycle_start_index,cycle_end_index);
	// Color cells
	
	 std:: vector<std::string> output(4, "black"); 
	if (pCell->phenotype.death.dead == false && pCell->type == 0) {
		int growth_color; 
		if (growth_rate > 1e-7) {
       			growth_color = (int) round((growth_rate/max_growth_rate)*155+100);
		} else { growth_color = 80;}
		char szTempString [128];
		sprintf( szTempString, "rgb(%u,%u,%u)",0 , 0, growth_color); 		
		output[0].assign( szTempString );
		output[1].assign( szTempString );
		output[2].assign( szTempString ); 
	} 
	
	if (pCell->phenotype.death.dead == false && pCell->type == 1) {
       		int growth_color_r;
		int growth_color_g;
	        if (growth_rate > 1e-7) {
		growth_color_r = (int) round((growth_rate/max_growth_rate)*155+100);
		growth_color_g = (int) round((growth_rate/max_growth_rate)*155+100);
		} else { growth_color_r = 80; growth_color_g = 80;}
		char szTempString [128];
		sprintf( szTempString, "rgb(%u,%u,%u)", growth_color_r, growth_color_g, 0); 		
		output[0].assign( szTempString );
		output[1].assign( szTempString );
		output[2].assign( szTempString ); 
	}
	// if cells are dead color differently
	if (pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::apoptotic )  // Apoptotic - Red
        {
                output[0] = "rgb(255,0,0)";
                output[2] = "rgb(125,0,0)";
        }

        // Necrotic - Brown
        if( pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_swelling ||
                pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_lysed ||
                pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic )
        {
                output[0] = "rgb(250,138,38)";
                output[2] = "rgb(139,69,19)";
        }
	

return output;} // paint_by_number_cell_coloring(pCell); }

void phenotype_function( Cell* pCell, Phenotype& phenotype, double dt )
{ return; }

/* Scuceptible cell rule */
void susceptible_cell_phenotype_update_rule( Cell* pCell, Phenotype& phenotype, double dt)
{
	update_cell_and_death_parameters_O2_based(pCell, phenotype, dt);
	if( phenotype.death.dead == true )
	{return;}
	
	bool div = pCell->phenotype.cycle.pCycle_Model->phases[0].division_at_phase_exit;

	// mutation
	if (phenotype.cycle.data.elapsed_time_in_phase < 1 && PhysiCell_globals.current_time > 10 ) {
		int ind_resistant = 1;
		if (UniformRandom() < parameters.doubles("mutation_rate")/2){
			pCell->convert_to_cell_definition(*cell_definitions_by_index[ind_resistant]);
			parameters.ints("number_of_denovo_mutations") += 1;
			pCell->clone_ID = parameters.ints("number_of_denovo_mutations");
			// std::bitset<128> temp_bitset = pCell->barcode;
	        // temp_bitset.set( 3*(pCell->number_of_divisions-1)+2+pCell->custom_data["left_most_bit"], 1 );
	        // pCell->barcode = temp_bitset;
		}
	}
	
	// set up parameters
	static int drug_index = microenvironment.find_density_index( "drug" );
	static int start_phase_index;
	static int end_phase_index;
	double treatment_drug_proliferation_saturation = 10.0;
    	double treatment_drug_death_saturation = parameters.doubles("drug_death_saturation");
   	double treatment_drug_death_threshold = parameters.doubles("drug_death_threshold");

	static int apoptosis_model_index = phenotype.death.find_death_model_index(PhysiCell_constants::apoptosis_death_model);

	double pDrug = (pCell->nearest_density_vector())[drug_index];
	double multiplier = 1.0;

	// increase apoptosis 
	if (phenotype.cycle.data.transition_rate(start_phase_index, end_phase_index)>0.1*pCell->parameters.pReference_live_phenotype->cycle.data.transition_rate(start_phase_index,end_phase_index)){
	if( pDrug > treatment_drug_death_threshold )
	{
		multiplier = parameters.doubles("treatment_strength")*( pDrug - treatment_drug_death_threshold )
				/ ( treatment_drug_death_saturation - treatment_drug_death_threshold );
	}

	if (pDrug > treatment_drug_death_saturation) {
			multiplier = parameters.doubles("treatment_strength");
			}

	phenotype.death.rates[apoptosis_model_index] = multiplier * pCell->parameters.pReference_live_phenotype->death.rates[apoptosis_model_index];
    	}

	return;

}

void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ return; } 

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt )
{ return; } 

std::string get_relevant_cell_info() {
    // try to change cell position to string;
				std::string data{"<Cells> \n"};
				std::string IDs{"ID: "};
				std::string pos_x{"x: "};
				std::string pos_y{"y: "};
				std::string pos_z{"z: "};
				std::string barcode{"barcode: "};
				std::string cell_type{"type: "};
				std::string elapsed_time_in_phase{"elapsed_time_in_phase: "};

					for (int cells_it = 0; cells_it < (*all_cells).size(); cells_it++) {
					    IDs.append(std::to_string((*all_cells)[cells_it]->ID));
					    IDs.append(",");
					    pos_x.append(std::to_string((*all_cells)[cells_it]->position[0]));
					    pos_x.append(",");
					    pos_y.append(std::to_string((*all_cells)[cells_it]->position[1]));
					    pos_y.append(",");
					    pos_z.append(std::to_string((*all_cells)[cells_it]->position[2]));
					    pos_z.append(",");
					    barcode.append((*all_cells)[cells_it]->barcode.to_string());
//                      barcode.append(std::to_string((*all_cells)[cells_it]->parent_ID));
					    barcode.append(",");
					    cell_type.append(std::to_string((*all_cells)[cells_it]->type));
					    cell_type.append(",");
					    elapsed_time_in_phase.append(std::to_string((*all_cells)[cells_it]->phenotype.cycle.data.elapsed_time_in_phase));
					    elapsed_time_in_phase.append(",");

				}
                data.append(IDs);
                data.append(";");
                data.append(pos_x);
                data.append(";");
                data.append(pos_y);
                data.append(";");
                data.append(pos_z);
                data.append(";");
                data.append(barcode);
                data.append(";");
                data.append(cell_type);
                data.append(";");
                data.append(elapsed_time_in_phase);
                data.append(";");
                data.append("end:");

return data;
}