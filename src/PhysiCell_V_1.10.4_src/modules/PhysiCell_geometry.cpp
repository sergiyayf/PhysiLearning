#include "./PhysiCell_geometry.h"

namespace PhysiCell{

// square fills 

void fill_rectangle( std::vector<double> bounds , Cell_Definition* pCD , double compression )
{
	double cell_radius = pCD->phenotype.geometry.radius; 
	double spacing = compression * cell_radius * 2.0; 
	double half_space = 0.5*spacing; 
	double y_offset = sqrt(3.0)*half_space; 
	
	// bounds? [xmin,ymin, zmin , xmax,ymax, zmax] 
	// assume z = 0.5*(zmin+zmax) 
	double Xmin; 
	double Xmax; 
	double Ymin; 
	double Ymax; 
	double Zmin; 
	double Zmax; 
	if( bounds.size() == 4 ) // only gave xmin,ymin,xmax,ymax 
	{
		Xmin = bounds[0]; 
		Ymin = bounds[1]; 
		
		Xmax = bounds[2]; 
		Ymax = bounds[3]; 
		
		Zmin = 0.0; 
		Zmax = 0.0; 
	}
	else
	{
		Xmin = bounds[0]; 
		Ymin = bounds[1]; 
		Zmin = bounds[2]; 
		
		Xmax = bounds[3]; 
		Ymax = bounds[4]; 
		Zmax = bounds[5]; 
	}
	
	double x = Xmin + cell_radius; 
	double y = Ymin + cell_radius; 
	double z = 0.5*( Zmin + Zmax ); 
	
	int n = 0; 
	
	while( y <= Ymax - cell_radius )
	{
		while( x <= Xmax - cell_radius )
		{
			Cell* pC = create_cell( *pCD ); 
			pC->assign_position( x,y,z ); 
			
			x += spacing; 
		}
		x = Xmin + half_space; 
		
		n++; 
		y += y_offset; 
		if( n % 2 == 1 )
		{ x += half_space; }
	}
	return; 
}

void fill_rectangle( std::vector<double> bounds , Cell_Definition* pCD )
{ return fill_rectangle(bounds,pCD,1.0); } 

void fill_rectangle( std::vector<double> bounds , int cell_type , double compression )
{ return fill_rectangle(bounds,find_cell_definition(cell_type),compression); }

void fill_rectangle( std::vector<double> bounds , int cell_type )
{ return fill_rectangle(bounds,find_cell_definition(cell_type),1.0); }

// circle fills 

void fill_circle( std::vector<double> center , double radius , Cell_Definition* pCD , double compression )
{
	double cell_radius = pCD->phenotype.geometry.radius; 
	double spacing = compression * cell_radius * 2.0; 
	double half_space = 0.5*spacing; 
	double y_offset = sqrt(3.0)*half_space; 
	
	double r_m_cr_2 = (radius-cell_radius)*(radius-cell_radius);  
	
	double Xmin = center[0] - radius; 
	double Xmax = center[0] + radius; 

	double Ymin = center[1] - radius; 
	double Ymax = center[1] + radius; 
	
	double x = Xmin + cell_radius; 
	double y = Ymin + cell_radius; 
	double z = center[2]; 
	
	int n = 0; 
	
	while( y <= Ymax - cell_radius )
	{
		while( x <= Xmax - cell_radius )
		{
			double d2 = (center[0]-x)*(center[0]-x) + (center[1]-y)*(center[1]-y); 
			// if we're within the circle, accept position and lay the cell 
			// essentially, we are applying a circular mask 
			if( d2 <= r_m_cr_2 )
			{
				Cell* pC = create_cell( *pCD ); 
				pC->assign_position( x,y,z ); 
			}
			x += spacing; 
		}
		y += y_offset; 
		n++; 
		
		x = Xmin+cell_radius;
		if( n % 2 == 1 )
		{ x += half_space; }
	}
	return; 
}

void fill_circle( std::vector<double> center , double radius , Cell_Definition* pCD )
{ return fill_circle( center,radius,pCD,1.0); } 

void fill_circle( std::vector<double> center , double radius , int cell_type , double compression )
{ return fill_circle( center,radius,find_cell_definition(cell_type),compression); } 

void fill_circle( std::vector<double> center , double radius , int cell_type ) 
{ return fill_circle( center,radius,find_cell_definition(cell_type),1); } 

// annulus 

void fill_annulus( std::vector<double> center , double outer_radius, double inner_radius , Cell_Definition* pCD , double compression )
{
	double cell_radius = pCD->phenotype.geometry.radius; 
	double spacing = compression * cell_radius * 2.0; 
	double half_space = 0.5*spacing; 
	double y_offset = sqrt(3.0)*half_space; 
	
	double ro_m_cr_2 = (outer_radius-cell_radius)*(outer_radius-cell_radius);  
	double ri_p_cr_2 = (inner_radius+cell_radius)*(inner_radius+cell_radius);  
	
	double Xmin = center[0] - outer_radius; 
	double Xmax = center[0] + outer_radius; 

	double Ymin = center[1] - outer_radius; 
	double Ymax = center[1] + outer_radius; 
	
	double x = Xmin + cell_radius; 
	double y = Ymin + cell_radius; 
	double z = center[2]; 
	
	int n = 0; 
	
	while( y <= Ymax - cell_radius )
	{
		while( x <= Xmax - cell_radius )
		{
			double d2 = (center[0]-x)*(center[0]-x) + (center[1]-y)*(center[1]-y); 
			// if we're within the circle, accept position and lay the cell 
			// essentially, we are applying a circular mask 
			if( d2 <= ro_m_cr_2 && d2 >= ri_p_cr_2 )
			{
				Cell* pC = create_cell( *pCD ); 
				pC->assign_position( x,y,z ); 
			}
			x += spacing; 
		}
		y += y_offset; 
		n++; 
		
		x = Xmin+cell_radius;
		if( n % 2 == 1 )
		{ x += half_space; }
	}
	return; 
}

void fill_annulus( std::vector<double> center , double outer_radius , double inner_radius, Cell_Definition* pCD )
{ return fill_annulus( center,outer_radius,inner_radius,pCD,1.0); } 

void fill_annulus( std::vector<double> center , double outer_radius , double inner_radius, int cell_type , double compression )
{ return fill_annulus( center,outer_radius,inner_radius,find_cell_definition(cell_type),1.0); } 

void fill_annulus( std::vector<double> center , double outer_radius , double inner_radius, int cell_type ) 
{ return fill_annulus( center,outer_radius,inner_radius,find_cell_definition(cell_type),1.0); } 

// draw lines 

void draw_line( std::vector<double> start , std::vector<double> end , Cell_Definition* pCD , double compression )
{
	double cell_radius = pCD->phenotype.geometry.radius; 
	double cr2 = cell_radius * cell_radius; 
	double spacing = compression * cell_radius * 2.0; 
	
	std::vector<double> position = start; 
	
	std::vector<double> displacement = end-position; 
	
	// get direction 
	std::vector<double> increment = displacement; 
	normalize( &increment ); // unit vector in correct direction along the line 
	increment *= spacing; // now it's the correct "delta" between cells along the line   
	
	double d2 = norm_squared( displacement ); 
	
	while( d2 > cr2 )
	{
		Cell* pC = create_cell( *pCD ); 
		pC->assign_position( position ); 
		
		position += increment; 
		displacement = end-position; 
		d2 = norm_squared( displacement ); 
	}
	return; 
}

void draw_line( std::vector<double> start , std::vector<double> end , Cell_Definition* pCD )
{ return draw_line(start,end,pCD,1.0); }

void draw_line( std::vector<double> start , std::vector<double> end , int cell_type , double compression )
{ return draw_line(start,end,find_cell_definition(cell_type),compression); }

void load_cells_csv( std::string filename )
{
	std::ifstream file( filename, std::ios::in );
	if( !file )
	{ 
		std::cout << "Error: " << filename << " not found during cell loading. Quitting." << std::endl; 
		exit(-1);
	}

	std::string line;
	while (std::getline(file, line))
	{
		std::vector<double> data;
		csv_to_vector( line.c_str() , data ); 

		if( data.size() != 4 )
		{
			std::cout << "Error! Importing cells from a CSV file expects each row to be x,y,z,typeID." << std::endl;
			exit(-1);
		}

		std::vector<double> position = { data[0] , data[1] , data[2] };

		int my_type = (int) data[3]; 
		Cell_Definition* pCD = find_cell_definition( my_type );
		if( pCD != NULL )
		{
			std::cout << "Creating " << pCD->name << " (type=" << pCD->type << ") at " 
			<< position << std::endl; 
			Cell* pCell = create_cell( *pCD ); 
			pCell->assign_position( position ); 
		}
		else
		{
			std::cout << "Warning! No cell definition found for index " << my_type << "!" << std::endl
			<< "\tIgnoring cell in " << filename << " at position " << position << std::endl; 
		}

	}

	file.close(); 	
}

bool load_cells_from_pugixml( pugi::xml_node root )
{
	pugi::xml_node node = root.child( "initial_conditions" ); 
	if( !node )
	{ 
		std::cout << "Warning: XML-based cell positions has wrong formating. Ignoring!" << std::endl; 
		return false;
	}

	node = node.child( "cell_positions" ); 
	if( !node )
	{
		std::cout << "Warning: XML-based cell positions has wrong formating. Ignoring!" << std::endl; 
		 return false;
	}

	// enabled? 
	if( node.attribute("enabled").as_bool() == false )
	{ return false; }

	// get filename 

	std::string folder = xml_get_string_value( node, "folder" ); 
	std::string filename = xml_get_string_value( node, "filename" ); 
	std::string input_filename = folder + "/" + filename; 

	std::string filetype = node.attribute("type").value() ; 

	// what kind? 
	if( filetype == "csv" || filetype == "CSV" )
	{
		std::cout << "Loading cells from CSV file " << input_filename << " ... " << std::endl; 
		load_cells_csv( input_filename );
		system("sleep 1");
		return true; 
	}
	if( filetype == "matlab" || filetype == "mat" || filetype == "MAT" )
	{
		std::cout << "Error: Load cell positions from matlab not yet supported. Try CSV." << std::endl; 
		exit(-1); 
		std::cout << "Loading cells from matlab file " << input_filename << " ... " << std::endl; 
		return false; 
	}
	if( filetype == "scene" )
	{
		std::cout << "Error: load cell positions from scene not yet supported. Try CSV." << std::endl; 
		exit(-1); 
		std::cout << "Loading cells from scene file " << input_filename << " ... " << std::endl; 
		return false; 
	}
	if( filetype == "physicell" || filetype == "PhysiCell" )
	{
		std::cout << "Error: load cell positions from PhysiCell snapshot not yet supported. Try CSV." << std::endl; 
		exit(-1); 
		std::cout << "Loading cells from PhysiCell file " << input_filename << " ... " << std::endl; 
		return false; 
	}

	return false; 
}

bool load_cells_from_pugixml( void )
{ return load_cells_from_pugixml( physicell_config_root ); }

void load_cells_physicell( std::string outputname ) {
    Cell* pC;
    pugi::xml_document checkpointing_document;
    pugi::xml_node physicell_checkpoint_root;
    std::string filename = outputname+".xml";
    
    checkpointing_document.load_file(filename.c_str());
        
    pugi::xml_parse_result result = checkpointing_document.load_file( filename.c_str());
    if( result.status != pugi::xml_parse_status::status_ok )
	{
		std::cout << "Error loading " << filename << "!" << std::endl; 
		
	} else {
        std::cout << "Succesfully loaded " << filename << "!" << std::endl; 
    }
    
    // Get the root of file    
    physicell_checkpoint_root = checkpointing_document.child("MultiCellDS");
    
    // Get name of the root node 
    std::string name = xml_get_my_name( physicell_checkpoint_root );
    
    // Now see if I can find the labels node; 
    
    pugi::xml_node cellular_information_node = physicell_checkpoint_root.child("cellular_information");
    pugi::xml_node cell_populations_node = cellular_information_node.child("cell_populations");
    pugi::xml_node cell_population_node = cell_populations_node.child("cell_population");
    pugi::xml_node custom_node = cell_population_node.child("custom");
    pugi::xml_node simplified_data_node = custom_node.child("simplified_data");
    pugi::xml_node labels_node = simplified_data_node.child("labels"); 
    std::string name2 = xml_get_my_name( labels_node );
        
    // Now loop through all labels node and see if I can find all the paramet names in the xml file
    pugi::xml_node node1 = labels_node.first_child(); 
		
    // Read matlab cell data into a matrix 
    std::string matlab_filename = outputname+"_cells.mat";
    std::vector<std::vector<double>> B = read_matlab( matlab_filename );
    
    // Get number of cells in the simulation
    int number_of_cells = B[0].size();
    
    // Loop through every cell and initiate it in the simulation
    for (int i = 0; i < number_of_cells; i++) { 
        // get cell definition and create a cell
        node1 = labels_node.first_child(); 
        int index;
        while (node1) {
            std::string value = xml_get_my_string_value( node1 );
            std::string index_str = node1.attribute("index").value();
            index = std::stoi(index_str);
            if (value == "cell_type") {
                
                break;
            } else {
                node1 = node1.next_sibling(); 
            }
        }
            Cell_Definition* pCD = cell_definitions_by_index[B[index][i]]; 
            pC = create_cell( *pCD ); 
            // go through all saved data
            node1 = labels_node.first_child(); 
            
            while( node1 )
            {
                std::string value = xml_get_my_string_value( node1 );
                std::string index_str = node1.attribute("index").value();
                index = std::stoi(index_str);
                std::string size = node1.attribute("size").value(); 
                // General 
               
	       	if (value == "ID") {
                    pC->ID =  B[index][i];
                }
		// If the code also includes parent_ID for lineage tracking uncomment       
		//else if (value == "parent_ID") {
                //    pC->parent_ID =  B[index][i];
                //}
	       	else if (value == "position") {
                    pC->assign_position({B[index][i],B[index+1][i],B[index+2][i]});
                    
                } else if (value == "total_volume") {
                    pC->set_total_volume(B[index][i]);
                    
                } else if (value == "cell_type") {
                    pC->type =  B[index][i];
                } else if (value == "cycle_model") {
                    if (B[index][i] == 0) {
                    pC->functions.cycle_model = Ki67_advanced;
                    pC->phenotype.cycle.sync_to_cycle_model( pC->functions.cycle_model);
                    
                    } else if (B[index][i] == 1) {
                    pC->functions.cycle_model = Ki67_basic;
                    pC->phenotype.cycle.sync_to_cycle_model( pC->functions.cycle_model);
                    
                    } else if (B[index][i] == 2) {
                    pC->functions.cycle_model = flow_cytometry_cycle_model;
                    pC->phenotype.cycle.sync_to_cycle_model( pC->functions.cycle_model);
                  
                    }  else if (B[index][i] == 5) {
                    pC->functions.cycle_model = live;
                    pC->phenotype.cycle.sync_to_cycle_model( pC->functions.cycle_model);
                    
                    } else if (B[index][i] == 6) {
                    pC->functions.cycle_model = flow_cytometry_separated_cycle_model;
                    pC->phenotype.cycle.sync_to_cycle_model( pC->functions.cycle_model);
                   
                    } else if (B[index][i] == 7) {
                    pC->functions.cycle_model = cycling_quiescent;
                    pC->phenotype.cycle.sync_to_cycle_model( pC->functions.cycle_model);
                    
                    } 
                } else if (value == "current_phase") {
                    
                    
                    /*
                    std::cout<<"code"<<std::endl;
                    std::cout<<pC->phenotype.cycle.current_phase().code<<std::endl;
                    std::cout<<"B:"<<B[index][i]<<std::endl;
                    while (pC->phenotype.cycle.current_phase().code != B[index][i]){
                        // check which cycle model this is and how many phases are there.
                        // change the transition rates to 9e99, advance the cycle and rewrite the real transition rates at the end from matlab
                        pC->phenotype.cycle.data.transition_rate(0,1)=9e99;
                        pC->phenotype.cycle.data.transition_rate(1,2)=9e99;
                        pC->phenotype.cycle.data.transition_rate(2,3)=9e99;
                        pC->phenotype.cycle.data.transition_rate(3,0)=9e99;
                        pC->phenotype.cycle.advance_cycle(pC,pC->phenotype, 0.01);
                        
                        std::cout<<"New code"<<std::endl;
                        std::cout<<pC->phenotype.cycle.current_phase().code<<std::endl;
                    }*/
                } else if (value == "elapsed_time_in_phase") {
                    pC->phenotype.cycle.data.elapsed_time_in_phase =  B[index][i]; 
                } else if (value == "nuclear_volume") {
                    pC->phenotype.volume.nuclear =  B[index][i];
                } else if (value == "cytoplasmic_volume") {
                    pC->phenotype.volume.cytoplasmic =  B[index][i];
                } else if (value == "fluid_fraction") {
                    pC->phenotype.volume.fluid_fraction =  B[index][i];
                } else if (value == "calcified_fraction") {
                    pC->phenotype.volume.calcified_fraction = B[index][i];
                } else if (value == "orientation") {
                    pC->state.orientation[0] =  B[index][i];
                    pC->state.orientation[1] =  B[index+1][i];
                    pC->state.orientation[2] =  B[index+2][i];
                } else if (value == "polarity") {
                    pC->phenotype.geometry.polarity =  B[index][i];
                } else if (value == "velocity") {
                    pC->velocity[0] =  B[index][i];
                    pC->velocity[1] =  B[index+1][i];
                    pC->velocity[2] =  B[index+2][i];
                } 
                // STATE 
                  else if (value == "pressure") {
                    pC->state.simple_pressure =  B[index][i]; 
                } else if (value == "number_of_nuclei") {
                    pC->state.number_of_nuclei =  B[index][i];
                } else if (value == "damage") {
                    pC->state.damage =  B[index][i];
                } else if (value == "total_attack_time") {
                    pC->state.total_attack_time =  B[index][i];
                } else if (value == "contact_with_basement_membrane"){
                    pC->state.contact_with_basement_membrane = B[index][i];
                } else if (value == "current_cycle_phase_exit_rate"){
                    int phase_index = pC->phenotype.cycle.data.current_phase_index; 
                    pC->phenotype.cycle.data.exit_rate(phase_index) =  B[index][i];
                } else if (value == "dead") {
                    pC->phenotype.death.dead = B[index][i];
                } else if (value == "current_death_model"){
                    pC->phenotype.death.current_death_model_index =  B[index][i];
                } else if (value == "death_rates"){
                    for (int death_models = 0; death_models<std::stoi(size); death_models++){
                        
                        double value =  B[index+death_models][i];
                        pC->phenotype.death.rates[death_models] = value;
                        
                    }
                } 
                // VOLUME
                  else if (value == "cytoplasmic_biomass_change_rate"){
                    pC->phenotype.volume.cytoplasmic_biomass_change_rate= B[index][i];
                } else if (value == "nuclear_biomass_change_rate"){
                    pC->phenotype.volume.nuclear_biomass_change_rate= B[index][i];
                } else if (value == "fluid_change_rate"){
                    pC->phenotype.volume.fluid_change_rate= B[index][i];
                } else if (value == "calcification_rate"){
                    pC->phenotype.volume.calcification_rate= B[index][i];
                } else if (value == "target_solid_cytoplasmic"){
                    pC->phenotype.volume.target_solid_cytoplasmic= B[index][i];
                } else if (value == "target_solid_nuclear"){
                    pC->phenotype.volume.target_solid_nuclear= B[index][i];
                } else if (value == "target_fluid_fraction"){
                    pC->phenotype.volume.target_fluid_fraction= B[index][i];
                } 
                // GEOMETRY
                  else if (value == "radius"){
                    pC->phenotype.geometry.radius= B[index][i];
                } else if (value == "nuclear_radius"){
                    pC->phenotype.geometry.nuclear_radius= B[index][i];
                } else if (value == "surface_area"){
                    pC->phenotype.geometry.surface_area= B[index][i];
                } 
                // MECHANICS
                  else if (value == "cell_cell_adhesion_strength"){
                    pC->phenotype.mechanics.cell_cell_adhesion_strength= B[index][i];
                } else if (value == "cell_BM_adhesion_strength"){
                    pC->phenotype.mechanics.cell_BM_adhesion_strength= B[index][i];
                } else if (value == "cell_cell_repulsion_strength"){
                    pC->phenotype.mechanics.cell_cell_repulsion_strength= B[index][i];
                } else if (value == "cell_BM_repulsion_strength"){
                    pC->phenotype.mechanics.cell_BM_repulsion_strength= B[index][i];
                } else if (value == "cell_adhesion_affinities"){
                      for (int cell_type_ID = 0; cell_type_ID<std::stoi(size); cell_type_ID++){
                        
                        double value =  B[index+cell_type_ID][i];
                        std::string behavior = "adhesive affinity to cell type "+std::to_string(cell_type_ID);
                        
                        set_single_behavior(pC,behavior,value); 
                        
                    }
                } else if (value == "relative_maximum_adhesion_distance"){
                    pC->phenotype.mechanics.relative_maximum_adhesion_distance= B[index][i];                    
                } else if (value == "maximum_number_of_attachments"){
                    pC->phenotype.mechanics.maximum_number_of_attachments= B[index][i];                    
                } else if (value == "attachment_elastic_constant"){
                    pC->phenotype.mechanics.attachment_elastic_constant= B[index][i];                    
                } else if (value == "attachment_rate"){
                    pC->phenotype.mechanics.attachment_rate= B[index][i];                    
                } else if (value == "detachment_rate"){
                    pC->phenotype.mechanics.detachment_rate= B[index][i];                    
                }
                // MOTILITY
                  else if (value == "is_motile"){
                    pC->phenotype.motility.is_motile=B[index][i];                    
                } else if (value == "persistence_time"){
                    pC->phenotype.motility.persistence_time= B[index][i];                    
                } else if (value == "migration_speed"){
                    pC->phenotype.motility.migration_speed= B[index][i];                    
                } else if (value == "migration_bias_direction"){
                   pC->phenotype.motility.migration_bias_direction[0] =  B[index][i];
                   pC->phenotype.motility.migration_bias_direction[1] =  B[index+1][i];
                   pC->phenotype.motility.migration_bias_direction[2] =  B[index+2][i];                    
                } else if (value == "migration_bias"){
                    pC->phenotype.motility.migration_bias= B[index][i];                    
                } else if (value == "motility_vector"){
                    pC->phenotype.motility.motility_vector[0] =  B[index][i];
                    pC->phenotype.motility.motility_vector[1] =  B[index+1][i];
                    pC->phenotype.motility.motility_vector[2] =  B[index+2][i];   
                } else if (value == "chemotaxis_index"){
                    pC->phenotype.motility.chemotaxis_index= B[index][i];                    
                } else if (value == "chemotaxis_direction"){
                    pC->phenotype.motility.chemotaxis_direction= B[index][i];                    
                } else if (value == "chemotactic_sensitivities"){
                    
                        pugi::xml_node variables_node = physicell_checkpoint_root.child("microenvironment").child("domain").child("variables");
                        pugi::xml_node substrate_node = variables_node.first_child();
                        for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){
                                                                        
                        std::string substrate_name = substrate_node.attribute("name").value(); 
                        double value =  B[index+num_substrates][i];
                        std::string behavior = "chemotactic response to "+substrate_name;
                        
                        set_single_behavior(pC,behavior,value);
                            
                        substrate_node = substrate_node.next_sibling();
                       
                    }
                                                              
                } 
                // SECRETION
                  else if (value == "secretion_rates"){
                        pugi::xml_node variables_node = physicell_checkpoint_root.child("microenvironment").child("domain").child("variables");
                        pugi::xml_node substrate_node = variables_node.first_child();
                    for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){
                                                                        
                        std::string substrate_name = substrate_node.attribute("name").value(); 
                        double value =  B[index+num_substrates][i];
                        std::string behavior = substrate_name+" secretion";
                        
                        set_single_behavior(pC,behavior,value);
                            
                        substrate_node = substrate_node.next_sibling();
                       
                    } 
                    
                } else if (value == "uptake_rates"){
                        pugi::xml_node variables_node = physicell_checkpoint_root.child("microenvironment").child("domain").child("variables");
                        pugi::xml_node substrate_node = variables_node.first_child();
                    for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){
                                                                        
                        std::string substrate_name = substrate_node.attribute("name").value(); 
                        double value =  B[index+num_substrates][i];
                        set_single_behavior(pC,substrate_name+" uptake",value);
                            
                        substrate_node = substrate_node.next_sibling();
                       
                    }
                } else if (value == "saturation_densities"){
                        pugi::xml_node variables_node = physicell_checkpoint_root.child("microenvironment").child("domain").child("variables");
                        pugi::xml_node substrate_node = variables_node.first_child();
                    for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){
                                                                        
                        std::string substrate_name = substrate_node.attribute("name").value(); 
                        double value =  B[index+num_substrates][i];
                        set_single_behavior(pC,substrate_name+" secretion saturation density",value);
                            
                        substrate_node = substrate_node.next_sibling();
                       
                    }
                } else if (value == "net_export_rates"){
                        pugi::xml_node variables_node = physicell_checkpoint_root.child("microenvironment").child("domain").child("variables");
                        pugi::xml_node substrate_node = variables_node.first_child();
                    for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){
                                                                        
                        std::string substrate_name = substrate_node.attribute("name").value(); 
                        double value =  B[index+num_substrates][i];
                        set_single_behavior(pC,substrate_name+" export",value);
                            
                        substrate_node = substrate_node.next_sibling();
                       
                    }
                } 
                // MOLECULAR
                  else if (value == "internalized_total_substrates"){
                        
                    for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){                                                                      
                        
                        double value =  B[index+num_substrates][i];
                        pC->phenotype.molecular.internalized_total_substrates[num_substrates] = value;
                        
                       
                    }
                } else if (value == "fraction_released_at_death"){
                        
                    for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){                                                                      
                        
                        double value =  B[index+num_substrates][i];
                        pC->phenotype.molecular.fraction_released_at_death[num_substrates] = value;
                        
                       
                    }
                } else if (value == "fraction_transferred_when_ingested"){
                        
                    for (int num_substrates = 0; num_substrates<std::stoi(size); num_substrates++){                                                                      
                        
                        double value =  B[index+num_substrates][i];
                        pC->phenotype.molecular.fraction_transferred_when_ingested[num_substrates] = value;
                                               
                    }
                }
                // INTERACTIONS
                  else if (value == "dead_phagocytosis_rate"){
                    pC->phenotype.cell_interactions.dead_phagocytosis_rate= B[index][i];                    
                } else if (value == "live_phagocytosis_rates"){
                    for (int cell_type_ID = 0; cell_type_ID<std::stoi(size); cell_type_ID++){
                        
                        double value =  B[index+cell_type_ID][i];
                        set_single_behavior(pC,"phagocytose cell type "+std::to_string(cell_type_ID),value); 
                        
                    }
                } else if (value == "attack_rates"){
                    for (int cell_type_ID = 0; cell_type_ID<std::stoi(size); cell_type_ID++){
                        
                        double value =  B[index+cell_type_ID][i];
                        set_single_behavior(pC,"attack cell type "+std::to_string(cell_type_ID),value); 
                        
                    }
                } else if (value == "damage_rate"){
                    pC->phenotype.cell_interactions.damage_rate= B[index][i]; 
                } else if (value == "fusion_rates"){
                    for (int cell_type_ID = 0; cell_type_ID<std::stoi(size); cell_type_ID++){
                        
                        double value =  B[index+cell_type_ID][i];
                        set_single_behavior(pC,"fuse to cell type "+std::to_string(cell_type_ID),value); 
                        
                    }
                }
                // TRANSFORMATIONS
                 
                  else if (value == "transformation_rates"){
                    for (int cell_type_ID = 0; cell_type_ID<std::stoi(size); cell_type_ID++){
                        
                        double value =  B[index+cell_type_ID][i];
                        set_single_behavior(pC,"transform to cell type "+std::to_string(cell_type_ID),value); 
                        
                    }
                } 
                // CUSTOM
                else {
                    
                    // Check if the value is in custom data
                    // Custom scalar
                    
                    
                    for( int jj=0 ; jj < pC->custom_data.variables.size(); jj++ )
                        {
                            std::string name = pC->custom_data.variables[jj].name; 
                            
                            if (value == name) {
                                pC->custom_data[value] = B[index][i];
                                 
                            }
                        }
                    // Custom vector 
                    for( int jj=0 ; jj < pC->custom_data.vector_variables.size(); jj++ )
                        {
                            std::string name = pC->custom_data.vector_variables[jj].name; 
                            if (value == name) {
                                for (int k = 0; k<std::stoi(size);k++){
                                pC->custom_data[value] = B[index+k][i]; 
                                }
                                node1 = node1.next_sibling(); 
                            }
                        }
                        
                    
                }
       
                                
                node1 = node1.next_sibling(); 
               
                
            }

    }
    
return;}

}; 
