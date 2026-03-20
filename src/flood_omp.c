/*
 * Simulation of rainwater flooding
 *
 * OpenMP version (Highly Optimized)
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<string.h>
#include<sys/time.h>
#include<omp.h>

#define PRECISION       1000000
#define FIXED(a)    ( (int)((a) * PRECISION) )
#define FLOATING(a) ( (float)(a) / PRECISION )
#define PRECISION_FIXED 1
#define PRECISION_FLOAT 2
#define SCENARIO_SIZE   30
#define SPILLAGE_FACTOR 2

#include "rng.c"

double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

#define CONTIGUOUS_CELLS 4
int displacements[CONTIGUOUS_CELLS][2] = {
    {-1,  0}, // Top
    { 1,  0}, // Bottom
    { 0, -1}, // Left
    { 0,  1}  // Right
};

#define COORD_SCEN2MAT_X( x )   ( x * columns / SCENARIO_SIZE )
#define COORD_SCEN2MAT_Y( y )   ( y * rows / SCENARIO_SIZE )
#define COORD_MAT2SCEN_X( c )   ( c * SCENARIO_SIZE / columns )
#define COORD_MAT2SCEN_Y( r )   ( r * SCENARIO_SIZE / rows )
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define accessMat( arr, exp1, exp2 )    arr[ (int)(exp1) * columns + (int)(exp2) ]
#define accessMat3D( arr, exp1, exp2, exp3 ) arr[ ((int)(exp1) * columns * depths) + ((int)(exp2) * depths) + (int)(exp3) ]  

float get_height( char scenario, int row, int col, int rows, int columns ) {
    float x_min, x_max, y_min, y_max;
    if ( scenario == 'M' ) { 
        x_min = -3.3; x_max = 5.1; y_min = -0.5; y_max = 8.8;
    } else { 
        x_min = -5.5; x_max = -3; y_min = -0.1; y_max = 4.2;
    }
    float x = x_min + ( (x_max - x_min) / columns ) * col;
    float y = y_min + ( (y_max - y_min) / rows ) * row;
    float height = -1 / ( x*x+1 ) + 2 / ( y*y+1 ) + 0.5 * sin( 5 * sqrt( x*x+y*y ) ) / sqrt( x*x+y*y) + (x+y) / 3 + sin( x )* cos( y ) + 0.4 * sin( 3*x+y )+ 0.25 * cos( 4*y+x );

    #define LOW_DAM_HEIGHT  -1.0
    #define HIGH_DAM_HEIGHT -0.4
    if ( scenario == 'D' && x <= -4.96 && x >= -5.0 ) {
        if ( height < HIGH_DAM_HEIGHT ) height = HIGH_DAM_HEIGHT;
    } else if ( scenario == 'd' && x <= -5.3 && x >= -5.34 ) {
        if ( height < LOW_DAM_HEIGHT ) height = LOW_DAM_HEIGHT;
    }

    if ( scenario == 'M' ) return height * 30 + 400;
    else return height * 20 + 100;
}

typedef struct {
    float x, y, radius, intensity, speed, angle;
    int active;
} Cloud_t;

Cloud_t cloud_init( Cloud_t cloud_model, float front_distance, float front_width, float front_depth, float front_direction, int rows, int cols, rng_t *rnd_state) {
    Cloud_t cloud;
    cloud.x = (float)rng_next_between( rnd_state, 0, front_width ) - front_width / 2;
    cloud.y = (float)rng_next_between( rnd_state, 0, front_depth ) - front_depth / 2;
    float opposite = front_direction + 180;
    float tmp_x = cloud.x, tmp_y = cloud.y;
    cloud.x = tmp_x * cos( opposite * M_PI / 180.0 ) - tmp_y * sin( opposite * M_PI / 180.0 );
    cloud.y = tmp_x * sin( opposite * M_PI / 180.0 ) + tmp_y * cos( opposite * M_PI / 180.0 );
    float x_center = front_distance * cos( opposite * M_PI / 180.0 ) + SCENARIO_SIZE / 2;
    float y_center = front_distance * sin( opposite * M_PI / 180.0 ) + SCENARIO_SIZE / 2;
    cloud.x += x_center;
    cloud.y += y_center;
    cloud.radius = (float)rng_next_between( rnd_state, cloud_model.radius / 2, cloud_model.radius );
    cloud.intensity = (float)rng_next_between( rnd_state, cloud_model.intensity / 2, cloud_model.intensity );
    cloud.speed = (float)rng_next_between( rnd_state, cloud_model.speed / 2, cloud_model.speed );
    cloud.angle = front_direction + (float)rng_next_between( rnd_state, 0, cloud_model.angle ) - cloud_model.angle / 2;
    cloud.active = 1;
    return cloud;
}

void show_usage( char *program_name ) {
    fprintf(stderr,"Usage: %s <args...>\n", program_name );
}

int main(int argc, char *argv[]) {
    #define NUM_FIXED_ARGS  17
    if (argc < NUM_FIXED_ARGS) { show_usage(argv[0]); exit(EXIT_FAILURE); }

    int rows = atoi( argv[1] ), columns = atoi( argv[2] );
    char ground_scenario = argv[3][0];
    float threshold = atof( argv[4] );
    int num_minutes = atoi( argv[5] );
    float ex_factor = atoi( argv[6] ), front_distance = atof( argv[7] );
    float front_width = atof( argv[8] ), front_depth = atof( argv[9] ), front_direction = atof( argv[10] );
    int num_clouds = atoi( argv[11] );
    Cloud_t cloud_model;
    cloud_model.radius = atof( argv[12] ); cloud_model.intensity = atof( argv[13] );
    cloud_model.speed = atof( argv[14] ); cloud_model.angle = atof( argv[15] );
    unsigned int seed_clouds = (unsigned int)atol( argv[16] );
    rng_t rnd_state = rng_new( seed_clouds );

    int num_clouds_arg = (argc-NUM_FIXED_ARGS) / 6;
    Cloud_t clouds_arg[ num_clouds_arg ];
    for (int idx = NUM_FIXED_ARGS; idx < argc; idx += 6 ) {
        int pos = ( idx-NUM_FIXED_ARGS ) / 6;
        clouds_arg[ pos ].x = atof( argv[idx] ); clouds_arg[ pos ].y = atof( argv[idx + 1] );
        clouds_arg[ pos ].radius = atof( argv[idx + 2] ); clouds_arg[ pos ].intensity = atof( argv[idx + 3] );
        clouds_arg[ pos ].speed = atof( argv[idx + 4] ); clouds_arg[ pos ].angle = atof( argv[idx + 5] );
    }

    double ttotal = cp_Wtime();

    int   *water_level;
    float *ground, *spillage_flag, *spillage_level, *spillage_from_neigh;
    Cloud_t *clouds;

    ground = (float *)malloc( sizeof(float) * rows * columns );
    water_level = (int *)calloc( rows * columns, sizeof(int) );
    spillage_flag = (float *)calloc( rows * columns, sizeof(float) );
    spillage_level = (float *)calloc( rows * columns, sizeof(float) );
    spillage_from_neigh = (float *)calloc( rows * columns * CONTIGUOUS_CELLS, sizeof(float) );
    clouds = (Cloud_t *)malloc( sizeof(Cloud_t) * (num_clouds + num_clouds_arg) );

    #pragma omp parallel for schedule(static)
    for (int row_pos = 0; row_pos < rows; row_pos++) {
        for (int col_pos = 0; col_pos < columns; col_pos++) {
            accessMat(ground, row_pos, col_pos) = get_height(ground_scenario, row_pos, col_pos, rows, columns);
        }
    }

    for (int cloud = 0; cloud < num_clouds; cloud++) {
        clouds[cloud] = cloud_init(cloud_model, front_distance, front_width, front_depth, front_direction, rows, columns, &rnd_state);
    }
    for (int cloud = 0; cloud < num_clouds_arg; cloud++) {
        clouds[num_clouds + cloud] = clouds_arg[cloud];
    }
    num_clouds += num_clouds_arg;

    float max_water_scenario = 0.0;
    double max_spillage_iter = FLT_MAX, max_spillage_scenario = 0.0;
    int max_spillage_minute = 0;
    long total_water = 0, total_water_loss = 0, total_rain = 0;

    int minute;
    for (minute = 0; minute < num_minutes && max_spillage_iter > threshold; minute++)  {

        // 4.1. Clouds movement
        #pragma omp parallel for schedule(static)
        for (int cloud = 0; cloud < num_clouds; cloud++) {
            float km_minute = clouds[cloud].speed / 60;
            clouds[cloud].x += km_minute * cos( clouds[cloud].angle * M_PI / 180.0 );
            clouds[cloud].y += km_minute * sin( clouds[cloud].angle * M_PI / 180.0 );
        }

        // 4.2. Rainfall
        #pragma omp parallel for reduction(+:total_rain) schedule(dynamic)
        for (int cloud = 0; cloud < num_clouds; cloud++) {
            float row_start = COORD_SCEN2MAT_Y( MAX(0, clouds[cloud].y - clouds[cloud].radius ) );
            float row_end = COORD_SCEN2MAT_Y( MIN( clouds[cloud].y + clouds[cloud].radius, SCENARIO_SIZE ) );
            float col_start = COORD_SCEN2MAT_X( MAX(0, clouds[cloud].x - clouds[cloud].radius ) );
            float col_end = COORD_SCEN2MAT_X( MIN( clouds[cloud].x + clouds[cloud].radius, SCENARIO_SIZE ) );

            for (int row_pos = row_start; row_pos < row_end; row_pos++) {
                for (int col_pos = col_start; col_pos < col_end; col_pos++) {
                    float x_pos = COORD_MAT2SCEN_X( col_pos );
                    float y_pos = COORD_MAT2SCEN_Y( row_pos );
                    float distance = sqrt( pow( x_pos - clouds[cloud].x, 2 ) + pow( y_pos - clouds[cloud].y, 2 ) );
                    if (distance < clouds[cloud].radius) {
                        float rain = ex_factor * MAX(0, clouds[cloud].intensity - distance / clouds[cloud].radius * sqrt(clouds[cloud].intensity));
                        float meters_per_minute = rain / 1000 / 60;
                        
                        #pragma omp atomic
                        accessMat(water_level, row_pos, col_pos) += FIXED(meters_per_minute);
                        total_rain += FIXED(meters_per_minute);
                    }
                }
            }
        }
    
        // 4.3. Compute water spillage to neighbor cells
        #pragma omp parallel for reduction(+:total_water_loss) schedule(guided)
        for (int row_pos = 0; row_pos < rows; row_pos++) {
            for (int col_pos = 0; col_pos < columns; col_pos++) {
                if (accessMat(water_level, row_pos, col_pos) > 0) {
                    float sum_diff = 0, my_spillage_level = 0;
                    float current_height = accessMat(ground, row_pos, col_pos) + FLOATING(accessMat(water_level, row_pos, col_pos));
        
                    int depths = CONTIGUOUS_CELLS;
                    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                        int new_row = row_pos + displacements[cell_pos][0];
                        int new_col = col_pos + displacements[cell_pos][1];
                        float neighbor_height;

                        if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) 
                            neighbor_height = accessMat(ground, row_pos, col_pos);
                        else
                            neighbor_height = accessMat(ground, new_row, new_col) + FLOATING(accessMat(water_level, new_row, new_col));

                        if (current_height >= neighbor_height) {
                            float height_diff = current_height - neighbor_height;
                            sum_diff += height_diff;
                            my_spillage_level = MAX(my_spillage_level, height_diff);
                        }
                    }
                    my_spillage_level = MIN(FLOATING(accessMat(water_level, row_pos, col_pos)), my_spillage_level);   
                
                    if (sum_diff > 0.0) { 
                        float proportion = my_spillage_level / sum_diff;
                        if (proportion > 1e-8) {
                            accessMat(spillage_flag, row_pos, col_pos) = 1;
                            accessMat(spillage_level, row_pos, col_pos) = my_spillage_level;
                    
                            for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                                int new_row = row_pos + displacements[cell_pos][0];
                                int new_col = col_pos + displacements[cell_pos][1];

                                if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) {
                                    float neighbor_height = accessMat(ground, row_pos, col_pos);
                                    if (current_height >= neighbor_height) {
                                        total_water_loss += FIXED(proportion * (current_height - neighbor_height) / 2);
                                    }
                                } else {
                                    float neighbor_height = accessMat(ground, new_row, new_col) + FLOATING(accessMat(water_level, new_row, new_col));
                                    if (current_height >= neighbor_height) {
                                        accessMat3D(spillage_from_neigh, new_row, new_col, cell_pos) = proportion * (current_height - neighbor_height);
                                    }
                                }
                            }
                        }
                    }                   
                }
            }
        }

        // 4.5. Propagation of previously computed water spillage
        max_spillage_iter = 0.0;
        #pragma omp parallel for reduction(max:max_spillage_iter) reduction(max:max_spillage_scenario) schedule(guided)
        for (int row_pos = 0; row_pos < rows; row_pos++) {
            for (int col_pos = 0; col_pos < columns; col_pos++) {
                
                if (accessMat(spillage_flag, row_pos, col_pos) == 1) {
                    accessMat(water_level, row_pos, col_pos) -= FIXED(accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR);
                    
                    float curr_spill = accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR;
                    if (curr_spill > max_spillage_iter) max_spillage_iter = curr_spill;
                    
                    if (curr_spill > max_spillage_scenario) {
                        max_spillage_scenario = curr_spill;
                        max_spillage_minute = minute;
                    }
                }
                int depths = CONTIGUOUS_CELLS;
                for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                    accessMat(water_level, row_pos, col_pos) += FIXED(accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) / SPILLAGE_FACTOR);
                }
            }
        }

        // 4.6. Optimized Reset ancillary structures
        memset(spillage_flag, 0, rows * columns * sizeof(float));
        memset(spillage_level, 0, rows * columns * sizeof(float));
        memset(spillage_from_neigh, 0, rows * columns * CONTIGUOUS_CELLS * sizeof(float));
    }

    // 5. Statistics
    #pragma omp parallel for reduction(max:max_water_scenario) reduction(+:total_water) schedule(static)
    for (int row_pos = 0; row_pos < rows; row_pos++) {
        for (int col_pos = 0; col_pos < columns; col_pos++) {
            float water_float = FLOATING(accessMat(water_level, row_pos, col_pos));
            if (water_float > max_water_scenario) max_water_scenario = water_float;
            total_water += accessMat(water_level, row_pos, col_pos);
        }
    }

    free(ground); free(water_level); free(spillage_flag); free(spillage_level); free(spillage_from_neigh); free(clouds);

    ttotal = cp_Wtime() - ttotal;
    printf("\nTime: %lf\n", ttotal);
    printf("Result: %d, %d, %10.6lf, %10.6lf, %10.6lf, %10.6lf, %10.6f\n\n", 
            minute, max_spillage_minute, max_spillage_scenario, max_water_scenario, 
            FLOATING(total_rain), FLOATING(total_water), FLOATING(total_water_loss));
    printf("Check precision loss: %10.6f\n\n", FLOATING(total_rain - total_water - total_water_loss));

    return 0;
}