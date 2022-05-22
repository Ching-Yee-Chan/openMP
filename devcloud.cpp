#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>
using namespace std;

int main()
{
    double time0(0), time1(0);
    int is_cpu = true;
    #pragma omp target map(from:is_cpu, time0)
    {
        float matrix[1024][1024];
        for (int i = 0;i < 1024;i++)
        {
            for (int j = 0;j < i;j++)
                matrix[i][j] = 0;
            for(int j = i;j<1024;j++)
                matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
        }
        for (int k = 0;k < 1000;k++)
        {
            int row1 = rand() % 1024;
            int row2 = rand() % 1024;
            float mult = rand() & 1 ? 1 : -1;
            float mult2 = rand() & 1 ? 1 : -1;
            mult = mult2 * (rand() / double(RAND_MAX)) + mult;
            for (int j = 0;j < 1024;j++)
                matrix[row1][j] += mult * matrix[row2][j];
        }
        timeval tv_begin, tv_end;
        is_cpu = omp_is_initial_device();
        gettimeofday(&tv_begin, 0);
        for (int i = 0; i < 1024 - 1; i++) {
            for (int j = i + 1; j < 1024; j++) {
                matrix[i][j] = matrix[i][j] / matrix[i][i];
            }
            matrix[i][i] = 1;
            for (int k = i + 1; k < 1024; k++) {
                for (int j = i + 1; j < 1024; j++) {
                    matrix[k][j] = matrix[k][j] - matrix[i][j] * matrix[k][i];
                }
                matrix[k][i] = 0;
            }
        }
        gettimeofday(&tv_end, 0);
        time0 = ((long long)tv_end.tv_sec - (long long)tv_begin.tv_sec) * 1000.0 + ((long long)tv_end.tv_usec - (long long)tv_begin.tv_usec) / 1000.0;
    }
    #pragma omp target map(from:is_cpu, time0)
    {
        float matrix[1024][1024];
        for (int i = 0;i < 1024;i++)
        {
            for (int j = 0;j < i;j++)
                matrix[i][j] = 0;
            for(int j = i;j<1024;j++)
                matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
        }
        for (int k = 0;k < 1000;k++)
        {
            int row1 = rand() % 1024;
            int row2 = rand() % 1024;
            float mult = rand() & 1 ? 1 : -1;
            float mult2 = rand() & 1 ? 1 : -1;
            mult = mult2 * (rand() / double(RAND_MAX)) + mult;
            for (int j = 0;j < 1024;j++)
                matrix[row1][j] += mult * matrix[row2][j];
        }
        timeval tv_begin, tv_end;
        is_cpu = omp_is_initial_device();
        gettimeofday(&tv_begin, 0);
        int i, j, k; 
        #pragma omp parallel num_threads(8), private(i, j, k)
        for(k = 0; k < 1024; ++k){
            #pragma omp for
            for (j = k + 1;j < 1024;++j)
                matrix[k][j] = matrix[k][j] / matrix[k][k];
            matrix[k][k] = 1.0;
            #pragma omp for
            for (i = k + 1; i < 1024; ++i)
            {
                for (int j = i + 1; j < 1024; j++) {
                    matrix[k][j] = matrix[k][j] - matrix[i][j] * matrix[k][i];
                }
                matrix[k][i] = 0;
            }
        }
        gettimeofday(&tv_end, 0);
        time1 = ((long long)tv_end.tv_sec - (long long)tv_begin.tv_sec) * 1000.0 + ((long long)tv_end.tv_usec - (long long)tv_begin.tv_usec) / 1000.0;
    }
    std::cout << "Running on " << (is_cpu ? "CPU" : "GPU") << "\n";
    cout<<time0<<" "<<time1;
}
