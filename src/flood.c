/*
 * Simulation of rainwater flooding
 *
 * Reference sequential version (Do not modify this code)
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 *
 * v1.6
 *
 * (c) 2025 Arturo Gonzalez-Escribano and Diego Garcia Alvarez
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<sys/time.h>

/*
 * Water levels are stored with fixed preciseon
 * This avoids result differences when arithmetic operations are reordered
 */
#define PRECISION       1000000
#define	FIXED(a)	( (int)((a) * PRECISION) )
#define	FLOATING(a)	( (float)(a) / PRECISION )
#define	PRECISION_FIXED	1
#define	PRECISION_FLOAT	2

/*
 * Scenario size (km x km)
 */
#define SCENARIO_SIZE	30

/*
 * Spillage factor for equilibrium
 */
#define SPILLAGE_FACTOR	2

/*
 * Utils: Random generator
 */
#include "rng.c"

/* 
 * Utils: Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}


	/*
	*
	* START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	*
	*/

/*
 * Utils: Number of contiguous cells to consider for water spillage
 * 	0: up, 1: down, 2: left, 3: right
 * 	Displacements for the contiguous cells
 * 	This data structure can be changed and/or optimized by the students
 */
#define CONTIGUOUS_CELLS 4
int displacements[CONTIGUOUS_CELLS][2] = {
	{-1,  0}, // Top
	{ 1,  0}, // Bottom
	{ 0, -1}, // Left
	{ 0,  1}  // Right
};


/*
 * Utils: Macro-functions to transform coordinates, from scenario to matrix cells, and back
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define SCENARIO_SIZE	30
#define COORD_SCEN2MAT_X( x )	( x * columns / SCENARIO_SIZE )
#define COORD_SCEN2MAT_Y( y )	( y * rows / SCENARIO_SIZE )
#define COORD_MAT2SCEN_X( c )	( c * SCENARIO_SIZE / columns )
#define COORD_MAT2SCEN_Y( r )	( r * SCENARIO_SIZE / rows )

/*
 * Utils: Macro functions for the min and max of two numbers
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Utils: Macro function to simplify accessing data of 2D and 3D matrixes stored in a flattened array
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define accessMat( arr, exp1, exp2 )	arr[ (int)(exp1) * columns + (int)(exp2) ]
#define accessMat3D( arr, exp1, exp2, exp3 ) arr[ ((int)(exp1) * columns * depths) + ((int)(exp2) * depths) + (int)(exp3) ]  

/*
 * Function: Generate ground height for a given position
 * 	This function can be changed and/or optimized by the students
 */
float get_height( char scenario, int row, int col, int rows, int columns ) {
	// Choose scenario limits
	float x_min, x_max, y_min, y_max;
	if ( scenario == 'M' ) { // Mountains scenario
		x_min = -3.3;
		x_max = 5.1;
		y_min = -0.5;
		y_max = 8.8;
	}
	else { // Valley scenarios
		x_min = -5.5;
		x_max = -3;
		y_min = -0.1;
		y_max = 4.2;
	}

	// Compute scenario coordinates of the cell position
	float x = x_min + ( (x_max - x_min) / columns ) * col;
	float y = y_min + ( (y_max - y_min) / rows ) * row;

	// Compute function height
	float height = -1 / ( x*x+1 ) + 2 / ( y*y+1 ) + 0.5 * sin( 5 * sqrt( x*x+y*y ) ) / sqrt( x*x+y*y) + (x+y) / 3 + sin( x )* cos( y ) + 0.4 * sin( 3*x+y )+ 0.25 * cos( 4*y+x );

	// Substitute by the dam height in the proper scenarios
	#define LOW_DAM_HEIGHT	-1.0
	#define HIGH_DAM_HEIGHT	-0.4
	if ( scenario == 'D' && x <= -4.96 && x >= -5.0 ) {
		if ( height < HIGH_DAM_HEIGHT ) { 
			height = HIGH_DAM_HEIGHT;
		}
	}
	else if ( scenario == 'd' && x <= -5.3 && x >= -5.34 ) {
		if ( height < LOW_DAM_HEIGHT ) { 
			height = LOW_DAM_HEIGHT;
		}
	}

	// Transform to meters
	if ( scenario == 'M' ) 
		return height * 30 + 400;
	else
		return height * 20 + 100;
}

/* 
 * Structure to represent moving rainy clouds
 * 	This structure can be changed and/or optimized by the students
 */
typedef struct {
	float x;          // x coordinate of the center
	float y;          // y coordinate of the center
	float radius;     // radius of the cloud (km)
	float intensity;  // rainfall intensity (cm/h)
	float speed;      // speed of movement (km/h)
	float angle;      // angle of movement
	int active;       // active cloud
} Cloud_t;


/*
 * Function: Initialize cloud with random values
 * 	This function can be changed and/or optimized by the students
 */
Cloud_t cloud_init( Cloud_t cloud_model, float front_distance, float front_width, float front_depth, float front_direction, int rows, int cols, rng_t *rnd_state) {
	Cloud_t cloud;

	// Random position around the front center
	cloud.x = (float)rng_next_between( rnd_state, 0, front_width ) - front_width / 2;
	cloud.y = (float)rng_next_between( rnd_state, 0, front_depth ) - front_depth / 2;

	// Rotate
	float opposite = front_direction + 180;
	float tmp_x = cloud.x;
	float tmp_y = cloud.y;
	cloud.x = tmp_x * cos( opposite * M_PI / 180.0 ) - tmp_y * sin( opposite * M_PI / 180.0 );
	cloud.y = tmp_x * sin( opposite * M_PI / 180.0 ) + tmp_y * cos( opposite * M_PI / 180.0 );

	// Move center
	float x_center = front_distance * cos( opposite * M_PI / 180.0 ) + SCENARIO_SIZE / 2;
	float y_center = front_distance * sin( opposite * M_PI / 180.0 ) + SCENARIO_SIZE / 2;
	cloud.x += x_center;
	cloud.y += y_center;

	// Cloud random parameters
	cloud.radius = (float)rng_next_between( rnd_state, cloud_model.radius / 2, cloud_model.radius );
	cloud.intensity = (float)rng_next_between( rnd_state, cloud_model.intensity / 2, cloud_model.intensity );
	cloud.speed = (float)rng_next_between( rnd_state, cloud_model.speed / 2, cloud_model.speed );
	cloud.angle = front_direction + (float)rng_next_between( rnd_state, 0, cloud_model.angle ) - cloud_model.angle / 2;
	cloud.active = 1;
	return cloud;
}

	/*
	 *
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */


#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_matrix( int precision_type, int rows, int columns, void *mat, const char *msj ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty 
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
#ifndef ANIMATION
	printf("%s:\n",msj);
	printf("+");
	for( j=0; j<columns; j++ ) printf("----------");
	printf("+\n");
	printf("\n");
#endif
	// Y coordinates: Reversed, higher rows up
	for( i=rows-1; i>=0; i-- ) {
#ifndef ANIMATION
		printf("|");
#endif
		// X coordinates
		for( j=0; j<columns; j++ ) {
			if ( precision_type == PRECISION_FLOAT )
				printf(" %10.4f", accessMat( ((float *)mat), i, j ) );
			else
				printf(" %10.4f", FLOATING( accessMat( ((int *)mat), i, j ) ) );
		}
#ifndef ANIMATION
		printf("|\n");
#endif
		printf("\n");
	}
#ifndef ANIMATION
	printf("+");
	for( j=0; j<columns; j++ ) printf("----------");
	printf("+\n\n\n");
#else
	printf("\n");
#endif
}


/* 
 * Function: Print the current state of the clouds 
 */
void print_clouds(int num_clouds, Cloud_t *clouds) {
	/* 
	 * You don't need to optimize this function, it is only for pretty 
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	printf("Clouds:\n");
	for (int i = 0; i < num_clouds; i++) {
		printf("Cloud %d: x = %f, y = %f, radius = %f, intensity = %f, speed = %f, angle = %f\n", 
			i, 
			clouds[i].x,
			clouds[i].y,
			clouds[i].radius,
			clouds[i].intensity,
			clouds[i].speed,
			clouds[i].angle );
	}
	printf("\n");
}
#endif // DEBUG


/*
 * Function: Print the program usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"\nFlood Simulation - Simulate rain and flooding in %d x %d km^2\n", SCENARIO_SIZE, SCENARIO_SIZE );
	fprintf(stderr,"----------------------------------------------------------------\n");
	fprintf(stderr,"Usage: %s ", program_name );
        fprintf(stderr,"<rows> <columns> <ground_scenario(M|V|D|d)> <threshold> <num_minutes> <exaggeration_factor> <front_distance> <front_width> <front_depth> <front_direction(grad.)> <num_random_clouds> <cloud_max_radius(km)> <cloud_max_intensity(mm/h)> <cloud_max_speed(km/h)> <cloud_max_angle_aperture(grad.)> <clouds_rnd_seed>\n");
	fprintf(stderr,"\n");
        fprintf(stderr,"\tOptional arguments for special clouds: <cloud_start_x(km)> <cloud_start_y(km)> <cloud_radius(km)> <cloud_intensity(mm/h)> <cloud_speed(km/h)> <cloud_angle(grad.)> ...\n");
	fprintf(stderr,"\n");
        fprintf(stderr,"\tGround models: 'M' mountain lakes, 'V' valley, 'D' valley with high dam, 'd' valley with low dam\n");
        fprintf(stderr,"\tIntensity of rain (mm/h): Strong (15-30), Very strong (30-60), Torrential: Above 60\n");
        fprintf(stderr,"\n");
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
#ifdef DEBUG
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);
#endif

	#define NUM_FIXED_ARGS	17
	
	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
        if (argc < NUM_FIXED_ARGS) {
                fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
                show_usage( argv[0] );
                exit( EXIT_FAILURE );
        }
	if (argc > NUM_FIXED_ARGS) {
		if ( (argc-NUM_FIXED_ARGS) % 6 != 0 ) {
                	fprintf(stderr, "-- Error: Wrong number of arguments, there should be %d compulsory arguments + groups of 6 optional arguments\n", NUM_FIXED_ARGS );
                	exit( EXIT_FAILURE );
		}
	}

        /* 1.2. Read ground sizes and selection of ground scenario */
        int rows = atoi( argv[1] );
        int columns = atoi( argv[2] );
	char ground_scenario = argv[3][0];
	if ( ground_scenario != 'M' && ground_scenario != 'V' && ground_scenario != 'D' && ground_scenario != 'd' ) {
                fprintf(stderr, "-- Error: Wrong ground scenario\n\n");
                show_usage( argv[0] );
                exit( EXIT_FAILURE );
	}

        /* 1.3. Read termination conditions */
	float threshold = atof( argv[4] );
        int num_minutes = atoi( argv[5] );

        /* 1.4. Read clouds data */
        float ex_factor = atoi( argv[6] );
	float front_distance = atof( argv[7] );
	float front_width = atof( argv[8] );
	float front_depth = atof( argv[9] );
	float front_direction = atof( argv[10] );
	int num_clouds = atoi( argv[11] );
	Cloud_t cloud_model;
	cloud_model.x = cloud_model.y = 0;
	cloud_model.radius = atof( argv[12] );
	cloud_model.intensity = atof( argv[13] );
	cloud_model.speed = atof( argv[14] );
	cloud_model.angle = atof( argv[15] );
	cloud_model.active = 0;
	unsigned int seed_clouds = (unsigned int)atol( argv[16] );
	// Initialize random sequence
	rng_t rnd_state = rng_new( seed_clouds );

	/* 1.5. Read the non-random clouds information */
	int num_clouds_arg = (argc-NUM_FIXED_ARGS) / 6;
	Cloud_t clouds_arg[ num_clouds_arg ];
	int idx;
	for ( idx = NUM_FIXED_ARGS; idx < argc; idx += 6 ) {
		int pos = ( idx-NUM_FIXED_ARGS ) / 6;
		clouds_arg[ pos ].x = atof( argv[idx] );
		clouds_arg[ pos ].y = atof( argv[idx + 1] );
		clouds_arg[ pos ].radius = atof( argv[idx + 2] );
		clouds_arg[ pos ].intensity = atof( argv[idx + 3] );
		clouds_arg[ pos ].speed = atof( argv[idx + 4] );
		clouds_arg[ pos ].angle = atof( argv[idx + 5] );
	}

#ifdef DEBUG
#ifdef ANIMATION
	printf("%d %d\n", rows, columns);
	printf("%d\n", num_minutes + 1 );
#else
	/* 1.5. Print arguments */
	printf("Arguments, Num_minutes: %d\n", num_minutes);
	printf("Arguments, Rows: %d, Columns: %d\n", rows, columns);
	printf("Arguments, Groud scenario: %c\n", ground_scenario );
	printf("Arguments, Num_clouds: %d, Max_radius: %f, Max_intensity: %f, Max_speed: %f, Max_angle: %f, seed: %u\n",
		num_clouds,
		cloud_model.radius,
		cloud_model.intensity,
		cloud_model.speed,
		cloud_model.angle,
		seed_clouds );
	for( idx = 0; idx < num_clouds_arg; idx++ ) {
		printf("Arguments, Optional cloud %d: x: %f, y: %f, Radius: %f, Intensity: %f, Speed: %f, Angle: %f\n",
		idx,
		clouds_arg[idx].x,
		clouds_arg[idx].y,
		clouds_arg[idx].radius,
		clouds_arg[idx].intensity,
		clouds_arg[idx].speed,
		clouds_arg[idx].angle );
	}
	printf("\n");
#endif
#endif

	/* 2. Start global timer */
	double ttotal = cp_Wtime();


	/*
	*
	* START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	*
	*/


	int   *water_level;            // Level of water on each cell (fixed precision)
	float *ground;                 // Ground height
	float *spillage_flag;          // Indicates which cells are spilling to neighbors
	float *spillage_level;         // Maximum level of spillage of each cell
	float *spillage_from_neigh;    // Spillage from each neighbor
	Cloud_t *clouds;               // Clouds


	/* 3. Initialization */
	/* 3.1. Memory allocation */
	ground = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
	water_level = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	spillage_flag = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
	spillage_level = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
	spillage_from_neigh = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns * (size_t)CONTIGUOUS_CELLS);
	clouds = (Cloud_t *)malloc( sizeof(Cloud_t) * ( num_clouds + num_clouds_arg ) );

	if ( ground == NULL  
		|| water_level == NULL
		|| spillage_flag == NULL
		|| spillage_level == NULL
		|| spillage_from_neigh == NULL ) {
		fprintf(stderr,"-- Error allocating ground and rain structures for size: %d x %d \n", rows, columns );
		exit( EXIT_FAILURE );
	}
	if ( clouds == NULL ) {
		fprintf(stderr,"-- Error allocating clouds structures for size: %d\n", num_clouds );
		exit( EXIT_FAILURE );
	}

	/* 3.2. Ground generation and initialization of other structures */
	int row_pos, col_pos, depth_pos;
	for ( row_pos=0; row_pos<rows; row_pos++ ) {
		for ( col_pos=0; col_pos<columns; col_pos++ ) {
			accessMat( ground, row_pos, col_pos ) = get_height( ground_scenario, row_pos, col_pos, rows, columns );
			accessMat( water_level, row_pos, col_pos ) = 0;
			accessMat( spillage_flag, row_pos, col_pos ) = 0.0;
			accessMat( spillage_level, row_pos, col_pos ) = 0.0;
			int depths = CONTIGUOUS_CELLS;
			for ( depth_pos=0; depth_pos<depths; depth_pos++ ) 
				accessMat3D( spillage_from_neigh, row_pos, col_pos, depth_pos ) = 0.0;
		}
	}

#ifdef DEBUG
	print_matrix( PRECISION_FLOAT, rows, columns, ground, "Ground heights" );
#endif

	/* 3.3. Clouds initialization */
	/* 3.3.1 Random clouds generation */
	int cloud;
	for ( cloud = 0; cloud < num_clouds; cloud++ ) {
		clouds[ cloud ] = cloud_init( cloud_model, front_distance, front_width, front_depth, front_direction, rows, columns, &rnd_state);
	}
	/* 3.3.2 Copy optional argument clouds */
	for ( cloud = 0; cloud < num_clouds_arg; cloud++ ) 
		clouds[ num_clouds + cloud ] = clouds_arg[ cloud ];
	num_clouds += num_clouds_arg;

#ifdef DEBUG
#ifndef ANIMATION
	print_clouds(num_clouds, clouds);
#endif
#endif

	/* Initialize metrics */
	float max_water_scenario = 0.0;
	double max_spillage_iter = FLT_MAX;
	double max_spillage_scenario = 0.0;
	int max_spillage_minute = 0;
	// Metrics to acumulate fixed point values
	long total_water = 0;
	long total_water_loss = 0;
	long total_rain = 0;

	/* 4. Flood simulation (time iterations) */
	int minute;
	for ( minute=0; minute < num_minutes && max_spillage_iter > threshold; minute++ )  {

		int new_row, new_col;
		int cell_pos;

		/* 4.1. Clouds movement */
		for ( cloud = 0; cloud < num_clouds; cloud++ ) {
			// Calculate new position (x are rows, y are columns)
			float km_minute = clouds[cloud].speed / 60;
			clouds[cloud].x += km_minute * cos( clouds[cloud].angle * M_PI / 180.0 );
			clouds[cloud].y += km_minute * sin( clouds[cloud].angle * M_PI / 180.0 );
		}

#ifdef DEBUG
#ifndef ANIMATION
	print_clouds(num_clouds, clouds);
#endif
#endif

		/* 4.2. Rainfall */
		for ( cloud = 0; cloud < num_clouds; cloud++ ) {
			//Compute the bounding box area of the cloud
			float row_start = COORD_SCEN2MAT_Y( MAX(0, clouds[cloud].y - clouds[cloud].radius ) );
			float row_end = COORD_SCEN2MAT_Y( MIN( clouds[cloud].y + clouds[cloud].radius, SCENARIO_SIZE ) );
			float col_start = COORD_SCEN2MAT_X( MAX(0, clouds[cloud].x - clouds[cloud].radius ) );
			float col_end = COORD_SCEN2MAT_X( MIN( clouds[cloud].x + clouds[cloud].radius, SCENARIO_SIZE ) );
			float distance;

			// Add rain to the ground water level
			float row_pos, col_pos;
			for ( row_pos = row_start; row_pos < row_end; row_pos++ ) {
				for ( col_pos = col_start; col_pos < col_end; col_pos++ ) {
					float x_pos = COORD_MAT2SCEN_X( col_pos );
					float y_pos = COORD_MAT2SCEN_Y( row_pos );
					distance = sqrt( pow( x_pos - clouds[cloud].x, 2 ) + pow( y_pos - clouds[cloud].y, 2 ) );
					if ( distance < clouds[cloud].radius ) {
						float rain = ex_factor * MAX( 0, clouds[cloud].intensity - distance / clouds[cloud].radius * sqrt( clouds[cloud].intensity ) );
						float meters_per_minute = rain / 1000 / 60;
						accessMat( water_level, row_pos, col_pos ) += FIXED( meters_per_minute );
						total_rain += FIXED( meters_per_minute );
					}
				}
			}
		}

#ifdef DEBUG
	print_matrix( PRECISION_FIXED, rows, columns, water_level, "Water after rain" );
#endif
	
		/* 4.3. Compute water spillage to neighbor cells */
		for ( row_pos=0; row_pos<rows; row_pos++ ) {
			for ( col_pos=0; col_pos<columns; col_pos++ ) {
				if ( accessMat( water_level, row_pos, col_pos ) > 0 ) {
					float sum_diff = 0;
					float my_spillage_level = 0;

					/* 4.3.1. Differences between current-cell level and its neighbours  */
					float current_height = accessMat( ground, row_pos, col_pos ) + FLOATING( accessMat( water_level, row_pos, col_pos ) );
		
					// Iterate over the four neighboring cells using the displacement array
					for ( cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++ ) {
						new_row = row_pos + displacements[cell_pos][0];
						new_col = col_pos + displacements[cell_pos][1];
		
						float neighbor_height;

						// Check if the new position is within the matrix boundaries
						if ( new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns ) 
							// Out of borders: Same height as the cell with no water
							neighbor_height = accessMat( ground, row_pos, col_pos );
						else
							// Neighbor cell: Ground height + water level
							neighbor_height = accessMat( ground, new_row, new_col ) + FLOATING( accessMat( water_level, new_row, new_col ) );

						// Compute level differences
						if ( current_height >= neighbor_height ) {
							float height_diff = current_height - neighbor_height;
							sum_diff += height_diff;
							my_spillage_level = MAX( my_spillage_level, height_diff );
						}
					}
					my_spillage_level = MIN( FLOATING( accessMat( water_level, row_pos, col_pos ) ), my_spillage_level );	
				
					// Compute proportion of spillage to each neighbor
					if ( sum_diff > 0.0 ) { 
						float proportion = my_spillage_level / sum_diff;
						// If proportion is significative, spillage
						if ( proportion > 1e-8 ) {
							accessMat( spillage_flag, row_pos, col_pos ) = 1;
							accessMat( spillage_level, row_pos, col_pos ) = my_spillage_level;
					
							// Iterate over the four neighboring cells using the displacement array
							for ( cell_pos = 0; cell_pos < 4; cell_pos++) {
								new_row = row_pos + displacements[cell_pos][0];
								new_col = col_pos + displacements[cell_pos][1];
					
								float neighbor_height;

								// Check if the new position is within the matrix boundaries
								if ( new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns ) {
									// Spillage out of the borders: Water loss
									neighbor_height = accessMat( ground, row_pos, col_pos );
									if ( current_height >= neighbor_height ) {
										total_water_loss += FIXED( proportion * ( current_height - neighbor_height ) / 2 );
									}
								}
								else {
									// Spillage to a neighbor cell
									neighbor_height = accessMat( ground, new_row, new_col) + FLOATING( accessMat(water_level, new_row, new_col ) );
									if ( current_height >= neighbor_height ) {
										int depths = CONTIGUOUS_CELLS;
										accessMat3D( spillage_from_neigh, new_row, new_col, cell_pos ) = proportion * ( current_height - neighbor_height );
									}
								}
							}
						}
					}					
				}
			}
		}

		/* 4.5. Propagation of previuosly computer water spillage to/from neighbors */
		max_spillage_iter = 0.0;
		for ( row_pos=0; row_pos<rows; row_pos++ ) {
			for ( col_pos=0; col_pos<columns; col_pos++ ) {
				// If the cell has spillage
				if( accessMat( spillage_flag, row_pos, col_pos ) == 1 ) {

					// Eliminate the spillage from the origin cell
					accessMat( water_level, row_pos, col_pos ) -= FIXED( accessMat( spillage_level, row_pos, col_pos ) / SPILLAGE_FACTOR );

					// Compute termination condition: Maximum cell spillage during the iteration
					if ( accessMat( spillage_level, row_pos, col_pos ) / SPILLAGE_FACTOR > max_spillage_iter ) {
						max_spillage_iter = accessMat( spillage_level, row_pos, col_pos ) / SPILLAGE_FACTOR;
					}
					// Statistics: Record maximum cell spillage during the scenario and its time
					if ( accessMat( spillage_level, row_pos, col_pos ) / SPILLAGE_FACTOR > max_spillage_scenario ) {
						max_spillage_scenario = accessMat( spillage_level, row_pos, col_pos ) / SPILLAGE_FACTOR;
						max_spillage_minute = minute;
					}
				}

				// Accumulate spillage from neighbors
				for ( cell_pos= 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++ ) {
					int depths = CONTIGUOUS_CELLS;
					accessMat( water_level, row_pos, col_pos ) += FIXED( accessMat3D( spillage_from_neigh, row_pos, col_pos, cell_pos ) / SPILLAGE_FACTOR );
				}
			}
		}

#ifdef DEBUG
#ifndef ANIMATION
	print_matrix( PRECISION_FIXED, rows, columns, water_level, "Water after spillage" );
#endif
#endif

		/* 4.6. Reset ancillary structures */
		for ( row_pos=0; row_pos<rows; row_pos++ ) {
			for ( col_pos=0; col_pos<columns; col_pos++ ) {
				for ( cell_pos= 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++ ) {
					int depths = CONTIGUOUS_CELLS;
					accessMat3D( spillage_from_neigh, row_pos, col_pos, cell_pos ) = 0;
				}
				accessMat( spillage_flag, row_pos, col_pos ) = 0;
				accessMat( spillage_level, row_pos, col_pos ) = 0;
			}
		}
	}

	/* 5. Statistics: Total remaining water and maximum amount of water in a cell */
	max_water_scenario = 0.0;
	for ( row_pos = 0; row_pos < rows; row_pos++ ) {
		for ( col_pos = 0; col_pos < columns; col_pos++ ) {
			if ( FLOATING( accessMat( water_level, row_pos, col_pos ) ) > max_water_scenario )
				max_water_scenario = FLOATING( accessMat( water_level, row_pos, col_pos ) );
			total_water += accessMat( water_level, row_pos, col_pos );
		}
	}

	/* 6. Free resources */	
	free( ground );
	free( water_level );
	free( spillage_flag );
	free( spillage_level );
	free( spillage_from_neigh );
	free( clouds );


	/*
	 *
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */


	/* 7. Stop global time */
	ttotal = cp_Wtime() - ttotal;

#ifndef ANIMATION
	/* 8. Output for leaderboard */
	/* 8.1. Total computation time */
	printf( "\nTime: %lf\n", ttotal );

	/* 8.2. Results: Statistics */
	printf("Result: %d, %d, %10.6lf, %10.6lf, %10.6lf, %10.6lf, %10.6f\n\n", 
			minute,
			max_spillage_minute,
			max_spillage_scenario,
			max_water_scenario,
			FLOATING( total_rain ),
			FLOATING( total_water ),
			FLOATING( total_water_loss ) );
	printf("Check precision loss: %10.6f\n\n", FLOATING( total_rain - total_water - total_water_loss ) );
#else
	// This code line does not compute anything usefull
	// It is included to avoid a compilation warning in animation mode
	max_spillage_minute += 0;
#endif 

	/* 9. End */
	return 0;
}
