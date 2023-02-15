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

void create_cell_types( void )
{
	// set the random seed 
	SeedRandom( parameters.ints("random_seed") );  
	
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
	cell_defaults.functions.update_phenotype = NULL; // update_cell_and_death_parameters_O2_based; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = NULL; 
	
	cell_defaults.functions.add_cell_basement_membrane_interactions = NULL; 
	cell_defaults.functions.calculate_distance_to_membrane = NULL; 
	
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
	   Put any modifications to individual cell definitions here. 
	   
	   This is a good place to set custom functions. 
	*/ 
	
	cell_defaults.functions.update_phenotype = phenotype_function; 
	cell_defaults.functions.custom_cell_rule = NULL; 
	cell_defaults.functions.contact_function = contact_function; 
	
    static int drug_index = microenvironment.find_density_index( "drug" ); 
    
    Cell_Definition* pWT = find_cell_definition("susceptible");
    pWT->functions.custom_cell_rule = susceptible_cell_on_off_treatment_rule;
    pWT->functions.update_phenotype = phenotype_function;
    //pWT->phenotype.secretion.uptake_rates[drug_index] = 1.0; 
    
    Cell_Definition* pRT = find_cell_definition("resistant");
    pRT->functions.custom_cell_rule = NULL;
    pRT->functions.update_phenotype = phenotype_function;
    //pRT->phenotype.secretion.uptake_rates[drug_index] = 1.0; 
	/*
	   This builds the map of cell definitions and summarizes the setup. 
	*/
		
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
	std::vector<double> bc_vector = {38.0} ;
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
	 get_default_microenvironment()->set_substrate_dirichlet_activation(0,false);
	 
	 
}
void activate_drug_dc( void )
{
 
microenvironment.set_substrate_dirichlet_activation(0,true);

}
void deactivate_drug_dc( void )
{
microenvironment.set_substrate_dirichlet_activation(0,false); 
}

void treatment_on( void ) {
	parameters.bools("treatment") = true;
	return;
}
void treatment_off( void ) {
	parameters.bools("treatment") = false;
	return;
}

void setup_microenvironment( void )
{
	// set domain parameters 
	
	// put any custom code to set non-homogeneous initial conditions or 
	// extra Dirichlet nodes here. 
	
	// initialize BioFVM 
	
	initialize_microenvironment(); 	
	set_circular_boundary_conditions();
    
	return; 
}

void setup_2D_circular_tissue( void )
{
	// place a cluster of tumor cells at the center 
    double cell_radius = cell_defaults.phenotype.geometry.radius; 
	double cell_spacing = 0.95 * 2.0 * cell_radius; 
	
	
		
	Cell* pCell = NULL;
	    
    int n = 0; 
    int resistant_cells = parameters.ints("number_of_resistant_cells");
    int susceptible_cells = parameters.ints("number_of_susceptible_cells");
    
    double tumor_radius = std::sqrt(resistant_cells+susceptible_cells)*cell_radius; // 250.0; 
    double x = 0.0; 
    double x_outer = tumor_radius; 
	double y = 0.0; 
    double Xmin = microenvironment.mesh.bounding_box[0]; 
	double Ymin = microenvironment.mesh.bounding_box[1];
    double Xmax = microenvironment.mesh.bounding_box[3]; 
	double Ymax = microenvironment.mesh.bounding_box[4]; 
    double Xrange = Xmax - Xmin; 
	double Yrange = Ymax - Ymin;
    
	while( n < resistant_cells)
	{  
        
        double r = tumor_radius +1; 
        while (r>0.4*tumor_radius) {
        x = Xmin + UniformRandom()*Xrange; 
        y = Ymin + UniformRandom()*Yrange; 
        r = norm( {x,y,0.0} ); 
        }
        
        pCell = create_cell( get_cell_definition("resistant") );         
		pCell->assign_position( {x,y,0.0} );
		n++; 
	} 
	while( n < resistant_cells+susceptible_cells)
	{  
        
        double r = tumor_radius +1; 
        while (r>tumor_radius || r<0.4*tumor_radius ) {
        x = Xmin + UniformRandom()*Xrange; 
        y = Ymin + UniformRandom()*Yrange; 
        r = norm( {x,y,0.0} ); 
        }
        
        pCell = create_cell( get_cell_definition("susceptible") ); 
        pCell->assign_position( {x,y,0.0} );
		n++; 
	}  
	    
		
	
	return; 
}

void setup_tissue( void ) {

 	Cell* pCell = NULL; 	
        pCell = create_cell( get_cell_definition("susceptible") ); 
        pCell->assign_position( {0.0,0.0,0.0} );
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
       		int growth_color = (int) round((growth_rate/max_growth_rate)*155+100);
		char szTempString [128];
		sprintf( szTempString, "rgb(%u,%u,%u)",0 , 0, growth_color); 		
		output[0].assign( szTempString );
		output[1].assign( szTempString );
		output[2].assign( szTempString ); 
	} 
	
	if (pCell->phenotype.death.dead == false && pCell->type == 1) {
       		int growth_color = (int) round((growth_rate/max_growth_rate)*155+100);
		char szTempString [128];
		sprintf( szTempString, "rgb(%u,%u,%u)", growth_color, growth_color, 0); 		
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
{ 
    // if cell is dead, don't bother with future phenotype changes. 
	if( phenotype.death.dead == true )
	{
		pCell->functions.update_phenotype = NULL; 		
		return; 
	}
    static int cycle_start_index = live.find_phase_index( PhysiCell_constants::live ); 
	static int cycle_end_index = live.find_phase_index( PhysiCell_constants::live ); 
    double pressure = pCell->state.simple_pressure;
    
    double multiplier = 1.0;
    double growth_layer = 2; 
    multiplier = 1+1/(1+std::exp(growth_layer))-1/(1+std::exp(growth_layer-pressure));
    /*if (pressure < 3) {
        multiplier = 3 - pressure;
    } else {
        multiplier = 0.0;
    }*/
    phenotype.cycle.data.transition_rate(cycle_start_index,cycle_end_index) = multiplier * 
		pCell->parameters.pReference_live_phenotype->cycle.data.transition_rate(cycle_start_index,cycle_end_index);
        	 
    return; }

 /* Scuceptible cell rule */
void susceptible_cell_with_drug_treatment_rule( Cell* pCell, Phenotype& phenotype, double dt)
{
	if( phenotype.death.dead == true ) 
	{return;}
	
	// set up parameters 
	static int drug_index = microenvironment.find_density_index( "drug" ); 
	static int start_phase_index; 
	static int end_phase_index; 
	double treatment_drug_proliferation_saturation = 10.0; 
    	double treatment_drug_death_saturation = 30.0;
   	double treatment_drug_death_threshold = 15.0;
		
	static int apoptosis_model_index = phenotype.death.find_death_model_index(PhysiCell_constants::apoptosis_death_model); 
	
	double pDrug = (pCell->nearest_density_vector())[drug_index];
	double multiplier = 1.0; 
	if (phenotype.cycle.data.transition_rate(start_phase_index, end_phase_index)>0.8*pCell->parameters.pReference_live_phenotype->cycle.data.transition_rate(start_phase_index,end_phase_index)){
	if( pDrug > treatment_drug_death_threshold )
	{
		multiplier = 10000*( pDrug - treatment_drug_death_threshold ) 
				/ ( treatment_drug_death_saturation - treatment_drug_death_threshold );
	}
	
	if (pDrug > treatment_drug_death_saturation) {
			multiplier = 10000.0;  
			}
	
	phenotype.death.rates[apoptosis_model_index] = multiplier * pCell->parameters.pReference_live_phenotype->death.rates[apoptosis_model_index];  
    	}	
    
	return;		

}

 /* Scuceptible cell binary treatment rule */
void susceptible_cell_on_off_treatment_rule( Cell* pCell, Phenotype& phenotype, double dt)
{
	if( phenotype.death.dead == true ) 
	{return;}
	
	// set up parameters 
	
	static int start_phase_index; 
	static int end_phase_index; 
		
	static int apoptosis_model_index = phenotype.death.find_death_model_index(PhysiCell_constants::apoptosis_death_model); 
	
	double multiplier = 1.0;
        double relative_growth_rate = phenotype.cycle.data.transition_rate(start_phase_index, end_phase_index)/pCell->parameters.pReference_live_phenotype->cycle.data.transition_rate(start_phase_index,end_phase_index);
	if (relative_growth_rate > 0.95){
	if( parameters.bools("treatment") )
	{
		multiplier = std::exp(-relative_growth_rate)*55555000;
	}
	
	phenotype.death.rates[apoptosis_model_index] = multiplier * pCell->parameters.pReference_live_phenotype->death.rates[apoptosis_model_index];  
    	}	
    
	return;		

}
void custom_function( Cell* pCell, Phenotype& phenotype , double dt )
{ return; } 

void contact_function( Cell* pMe, Phenotype& phenoMe , Cell* pOther, Phenotype& phenoOther , double dt )
{ return; } 
