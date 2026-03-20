/*
 * Simulation of rainwater flooding
 *
 * MPI version (Distributed Memory - Optimized Ghost Rows)
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
#include<mpi.h>

#define PRECISION       1000000
#define FIXED(a)    ( (int)((a) * PRECISION) )
#define FLOATING(a) ( (float)(a) / PRECISION )
#define SCENARIO_SIZE   30
#define SPILLAGE_FACTOR 2

#include "rng.c"

double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

#define CONTIGUOUS_CELLS 4
int displacements[CONTIGUOUS_CELLS][2] = { {-1, 0}, { 1, 0}, { 0, -1}, { 0, 1} };

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
    if ( scenario == 'M' ) { x_min = -3.3; x_max = 5.1; y_min = -0.5; y_max = 8.8; }
    else { x_min = -5.5; x_max = -3; y_min = -0.1; y_max = 4.2; }
    float x = x_min + ( (x_max - x_min) / columns ) * col;
    float y = y_min + ( (y_max - y_min) / rows ) * row;
    float height = -1 / ( x*x+1 ) + 2 / ( y*y+1 ) + 0.5 * sin( 5 * sqrt( x*x+y*y ) ) / sqrt( x*x+y*y) + (x+y) / 3 + sin( x )* cos( y ) + 0.4 * sin( 3*x+y )+ 0.25 * cos( 4*y+x );

    if ( scenario == 'D' && x <= -4.96 && x >= -5.0 ) {
        if ( height < -0.4 ) height = -0.4;
    } else if ( scenario == 'd' && x <= -5.3 && x >= -5.34 ) {
        if ( height < -1.0 ) height = -1.0;
    }
    if ( scenario == 'M' ) return height * 30 + 400;
    else return height * 20 + 100;
}

typedef struct { float x, y, radius, intensity, speed, angle; int active; } Cloud_t;

Cloud_t cloud_init( Cloud_t cloud_model, float front_distance, float front_width, float front_depth, float front_direction, int rows, int cols, rng_t *rnd_state) {
    Cloud_t cloud;
    cloud.x = (float)rng_next_between( rnd_state, 0, front_width ) - front_width / 2;
    cloud.y = (float)rng_next_between( rnd_state, 0, front_depth ) - front_depth / 2;
    float opposite = front_direction + 180;
    float tmp_x = cloud.x, tmp_y = cloud.y;
    cloud.x = tmp_x * cos( opposite * M_PI / 180.0 ) - tmp_y * sin( opposite * M_PI / 180.0 );
    cloud.y = tmp_x * sin( opposite * M_PI / 180.0 ) + tmp_y * cos( opposite * M_PI / 180.0 );
    cloud.x += front_distance * cos( opposite * M_PI / 180.0 ) + SCENARIO_SIZE / 2;
    cloud.y += front_distance * sin( opposite * M_PI / 180.0 ) + SCENARIO_SIZE / 2;
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
    int mpi_rank, mpi_nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);

    #define NUM_FIXED_ARGS  17
    if (argc < NUM_FIXED_ARGS) {
        if(mpi_rank == 0) show_usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

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

    MPI_Barrier(MPI_COMM_WORLD);
    double ttotal = cp_Wtime();

    MPI_Comm mpi_active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_rank >= rows ? 0:1, mpi_rank, &mpi_active_comm);
    if (mpi_rank >= rows) { MPI_Finalize(); return 0; }
    
    int mpi_active_rank, mpi_active_nprocs;
    MPI_Comm_rank(mpi_active_comm, &mpi_active_rank);
    MPI_Comm_size(mpi_active_comm, &mpi_active_nprocs);

    int numRows = rows / mpi_active_nprocs;
    int resto = rows % mpi_active_nprocs;
    if (mpi_active_rank < resto) numRows++;
    float firstRow = numRows * mpi_active_rank + (mpi_active_rank < resto ? 0 : resto);
    float lastRow = firstRow + numRows;

    // Allocate memory with 2 ghost rows (0 and numRows+1)
    size_t mem_size = (size_t)(numRows + 2) * columns;
    int depths = CONTIGUOUS_CELLS;
    float *ground = (float *)malloc(mem_size * sizeof(float));
    int *water_level = (int *)calloc(mem_size, sizeof(int));
    float *spillage_flag = (float *)calloc(mem_size, sizeof(float));
    float *spillage_level = (float *)calloc(mem_size, sizeof(float));
    float *spillage_from_neigh = (float *)calloc(mem_size * CONTIGUOUS_CELLS, sizeof(float));
    Cloud_t *clouds = (Cloud_t *)malloc(sizeof(Cloud_t) * (num_clouds + num_clouds_arg));

    float *mandar_arriba = (float *)calloc(columns, sizeof(float));
    float *mandar_abajo = (float *)calloc(columns, sizeof(float));
    float *recibir_arriba = (float *)calloc(columns, sizeof(float));
    float *recibir_abajo = (float *)calloc(columns, sizeof(float));

    // Ground generation for local rows
    for (int row_pos = 1; row_pos <= numRows; row_pos++) {
        for (int col_pos = 0; col_pos < columns; col_pos++) {
            accessMat(ground, row_pos, col_pos) = get_height(ground_scenario, (row_pos + firstRow - 1), col_pos, rows, columns);
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

        for (int cloud = 0; cloud < num_clouds; cloud++) {
            float km_minute = clouds[cloud].speed / 60;
            clouds[cloud].x += km_minute * cos( clouds[cloud].angle * M_PI / 180.0 );
            clouds[cloud].y += km_minute * sin( clouds[cloud].angle * M_PI / 180.0 );
        }
        
        for (int cloud = 0; cloud < num_clouds; cloud++) {
            float row_start = COORD_SCEN2MAT_Y( MAX(0, clouds[cloud].y - clouds[cloud].radius ) );
            float row_end = COORD_SCEN2MAT_Y( MIN( clouds[cloud].y + clouds[cloud].radius, SCENARIO_SIZE ) );
            float col_start = COORD_SCEN2MAT_X( MAX(0, clouds[cloud].x - clouds[cloud].radius ) );
            float col_end = COORD_SCEN2MAT_X( MIN( clouds[cloud].x + clouds[cloud].radius, SCENARIO_SIZE ) );

            for (int row_pos = row_start; row_pos < row_end; row_pos++) {
                if (row_pos < firstRow || row_pos >= lastRow) continue;
                float local_row = row_pos - firstRow + 1;
                
                for (int col_pos = col_start; col_pos < col_end; col_pos++) {
                    float distance = sqrt( pow(COORD_MAT2SCEN_X(col_pos) - clouds[cloud].x, 2) + pow(COORD_MAT2SCEN_Y(row_pos) - clouds[cloud].y, 2) );
                    if (distance < clouds[cloud].radius) {
                        float rain = ex_factor * MAX( 0, clouds[cloud].intensity - distance / clouds[cloud].radius * sqrt( clouds[cloud].intensity ) );
                        float meters_per_minute = rain / 1000 / 60;
                        accessMat(water_level, local_row, col_pos) += FIXED(meters_per_minute);
                        total_rain += FIXED(meters_per_minute);
                    }
                }
            }
        }
        
        // --- GHOST ROWS EXCHANGE (Optimized non-blocking) ---
        MPI_Request reqs[8];
        int num_reqs = 0;

        // Receive directly into ghost rows
        if (mpi_active_rank > 0) {
            MPI_Irecv(&accessMat(water_level, 0, 0), columns, MPI_INT, mpi_active_rank - 1, 2, mpi_active_comm, &reqs[num_reqs++]);
            MPI_Irecv(&accessMat(ground, 0, 0), columns, MPI_FLOAT, mpi_active_rank - 1, 3, mpi_active_comm, &reqs[num_reqs++]);
        }
        if (mpi_active_rank < mpi_active_nprocs - 1) {
            MPI_Irecv(&accessMat(water_level, numRows + 1, 0), columns, MPI_INT, mpi_active_rank + 1, 0, mpi_active_comm, &reqs[num_reqs++]);
            MPI_Irecv(&accessMat(ground, numRows + 1, 0), columns, MPI_FLOAT, mpi_active_rank + 1, 1, mpi_active_comm, &reqs[num_reqs++]);
        }
        
        // Send actual boundary rows
        if (mpi_active_rank > 0) {
            MPI_Isend(&accessMat(water_level, 1, 0), columns, MPI_INT, mpi_active_rank - 1, 0, mpi_active_comm, &reqs[num_reqs++]);
            MPI_Isend(&accessMat(ground, 1, 0), columns, MPI_FLOAT, mpi_active_rank - 1, 1, mpi_active_comm, &reqs[num_reqs++]);
        }
        if (mpi_active_rank < mpi_active_nprocs - 1) {
            MPI_Isend(&accessMat(water_level, numRows, 0), columns, MPI_INT, mpi_active_rank + 1, 2, mpi_active_comm, &reqs[num_reqs++]);
            MPI_Isend(&accessMat(ground, numRows, 0), columns, MPI_FLOAT, mpi_active_rank + 1, 3, mpi_active_comm, &reqs[num_reqs++]);
        }
        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

        // 4.3 Compute spillage
        for (int row_pos = 1; row_pos <= numRows; row_pos++) {
            for (int col_pos = 0; col_pos < columns; col_pos++) {
                if (accessMat(water_level, row_pos, col_pos) > 0) {
                    float sum_diff = 0, my_spillage_level = 0;
                    float current_height = accessMat(ground, row_pos, col_pos) + FLOATING(accessMat(water_level, row_pos, col_pos));
        
                    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                        int new_row = row_pos + displacements[cell_pos][0];
                        int new_col = col_pos + displacements[cell_pos][1];
                        float neighbor_height;

                        if (new_col < 0 || new_col >= columns) {
                            neighbor_height = accessMat(ground, row_pos, col_pos);
                        } else {
                            if ((mpi_active_rank == 0 && new_row < 1) || (mpi_active_rank == mpi_active_nprocs - 1 && new_row > numRows)) {
                                neighbor_height = accessMat(ground, row_pos, col_pos);
                            } else {
                                neighbor_height = accessMat(ground, new_row, new_col) + FLOATING(accessMat(water_level, new_row, new_col));                              
                            }
                        }

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

                                if (new_col < 0 || new_col >= columns) {
                                    if (current_height >= accessMat(ground, row_pos, col_pos)) {
                                        total_water_loss += FIXED(proportion * (current_height - accessMat(ground, row_pos, col_pos)) / 2);
                                    }
                                } else {
                                    if ((mpi_active_rank == 0 && new_row < 1) || (mpi_active_rank == mpi_active_nprocs - 1 && new_row > numRows)) {
                                        if (current_height >= accessMat(ground, row_pos, col_pos)) {
                                            total_water_loss += FIXED(proportion * (current_height - accessMat(ground, row_pos, col_pos)) / 2);
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
        }

        // Exchange Spillage Arrays
        for (int col = 0; col < columns; col++) {
            mandar_arriba[col] = (lastRow < rows) ? accessMat3D(spillage_from_neigh, numRows + 1, col, 1) : 0.0;
            mandar_abajo[col] = (firstRow > 0) ? accessMat3D(spillage_from_neigh, 0, col, 0) : 0.0;
        }

        num_reqs = 0;
        if (mpi_active_rank > 0) {
            MPI_Irecv(recibir_arriba, columns, MPI_FLOAT, mpi_active_rank - 1, 0, mpi_active_comm, &reqs[num_reqs++]);
            MPI_Isend(mandar_abajo, columns, MPI_FLOAT, mpi_active_rank - 1, 1, mpi_active_comm, &reqs[num_reqs++]);
        }
        if (mpi_active_rank < mpi_active_nprocs - 1) {
            MPI_Irecv(recibir_abajo, columns, MPI_FLOAT, mpi_active_rank + 1, 1, mpi_active_comm, &reqs[num_reqs++]);
            MPI_Isend(mandar_arriba, columns, MPI_FLOAT, mpi_active_rank + 1, 0, mpi_active_comm, &reqs[num_reqs++]);
        }
        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

        for (int col = 0; col < columns; col++) {
            if (recibir_abajo[col] > 0) accessMat3D(spillage_from_neigh, numRows, col, 0) = recibir_abajo[col];
            if (recibir_arriba[col] > 0) accessMat3D(spillage_from_neigh, 1, col, 1) = recibir_arriba[col];
        }

        // 4.5. Propagation
        max_spillage_iter = 0.0;
        for (int row_pos = 1; row_pos <= numRows; row_pos++) {
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
                for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                    accessMat(water_level, row_pos, col_pos) += FIXED(accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) / SPILLAGE_FACTOR);
                }
            }
        }

        double spillage_iter_max, spillage_scenario_max;
        MPI_Allreduce(&max_spillage_iter, &spillage_iter_max, 1, MPI_DOUBLE, MPI_MAX, mpi_active_comm);
        max_spillage_iter = spillage_iter_max;

        MPI_Allreduce(&max_spillage_scenario, &spillage_scenario_max, 1, MPI_DOUBLE, MPI_MAX, mpi_active_comm);
        if(max_spillage_scenario < spillage_scenario_max) {
            max_spillage_scenario = spillage_scenario_max;
            max_spillage_minute = minute;
        }

        // 4.6. Reset ancillary structures (Insanely fast with memset)
        memset(spillage_flag, 0, mem_size * sizeof(float));
        memset(spillage_level, 0, mem_size * sizeof(float));
        memset(spillage_from_neigh, 0, mem_size * CONTIGUOUS_CELLS * sizeof(float));
    }

    max_water_scenario = 0.0;
    for (int row_pos = 1; row_pos <= numRows; row_pos++) {
        for (int col_pos = 0; col_pos < columns; col_pos++) {
            float wf = FLOATING(accessMat(water_level, row_pos, col_pos));
            if (wf > max_water_scenario) max_water_scenario = wf;
            total_water += accessMat(water_level, row_pos, col_pos);
        }
    }

    float rec_max_water_scenario;
    MPI_Reduce(&max_water_scenario, &rec_max_water_scenario, 1, MPI_FLOAT, MPI_MAX, 0, mpi_active_comm);
    
    long rec_total_rain, rec_total_water, rec_total_water_loss;
    MPI_Reduce(&total_rain, &rec_total_rain, 1, MPI_LONG, MPI_SUM, 0, mpi_active_comm);
    MPI_Reduce(&total_water, &rec_total_water, 1, MPI_LONG, MPI_SUM, 0, mpi_active_comm);
    MPI_Reduce(&total_water_loss, &rec_total_water_loss, 1, MPI_LONG, MPI_SUM, 0, mpi_active_comm);

    if (mpi_active_rank == 0) {
        max_water_scenario = rec_max_water_scenario;
        total_rain = rec_total_rain;
        total_water = rec_total_water;
        total_water_loss = rec_total_water_loss;
    }
    
    free(ground); free(water_level); free(spillage_flag); free(spillage_level); free(spillage_from_neigh); free(clouds);
    free(mandar_arriba); free(mandar_abajo); free(recibir_arriba); free(recibir_abajo);

    ttotal = cp_Wtime() - ttotal;

    if (mpi_rank == 0) {
        printf("\nTime: %lf\n", ttotal);
        printf("Result: %d, %d, %10.6lf, %10.6lf, %10.6lf, %10.6lf, %10.6f\n\n", 
                minute, max_spillage_minute, max_spillage_scenario, max_water_scenario, 
                FLOATING(total_rain), FLOATING(total_water), FLOATING(total_water_loss));
        printf("Check precision loss: %10.6f\n\n", FLOATING(total_rain - total_water - total_water_loss));
    }

    MPI_Finalize();
    return 0;
}