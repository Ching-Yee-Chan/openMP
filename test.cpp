#include <iostream>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX
#include <omp.h>
#include <stdio.h>
#include <windows.h>
using namespace std;
typedef long long ll;

#define ROW 1024 
#define TASK 10
#define INTERVAL 10000
float matrix[ROW][ROW];
int NUM_THREADS = 8;

//动态线程分配：待分配行数
int remain = ROW;

void init()
{
	for (int i = 0;i < ROW;i++)
	{
		for (int j = 0;j < i;j++)
			matrix[i][j] = 0;
		for(int j = i;j<ROW;j++)
			matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0;k < 8000;k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0;j < ROW;j++)
			matrix[row1][j] += mult * matrix[row2][j];
	}
}

void plain() {
	for (int i = 0; i < ROW - 1; i++) {
		for (int j = i + 1; j < ROW; j++) {
			matrix[i][j] = matrix[i][j] / matrix[i][i];
		}
		matrix[i][i] = 1;
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				matrix[k][j] = matrix[k][j] - matrix[i][j] * matrix[k][i];
			}
			matrix[k][i] = 0;
		}
	}
}

void SIMD()
{
    for(int k = 0; k < ROW; ++k)
	{
		__m128 diver = _mm_load_ps1(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i += 1)
		{ 
			__m128 mult1 = _mm_load_ps1(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				__m128 sub1 =  _mm_loadu_ps(&matrix[i][j]);
				__m128 mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
	}
}


void ver0()//动态限制
{
	int i, j, k; 
	__m128 mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    for(k = 0; k < ROW; ++k){
        // 串行部分，也可以尝试并行化
        __m128 diver = _mm_load_ps1(&matrix[k][k]);
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
		    matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
		    __m128 divee =  _mm_loadu_ps(&matrix[k][j]);
		    divee = _mm_div_ps(divee, diver);
		    _mm_storeu_ps(&matrix[k][j], divee);
	    }
	    matrix[k][k] = 1.0;
        // 并行部分，使用行划分
        #pragma omp parallel for num_threads(NUM_THREADS) 
        for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				sub1 =  _mm_loadu_ps(&matrix[i][j]);
				mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver1()//静态朴素 
{
	int i, j, k; 
	__m128 mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
    for(k = 0; k < ROW; ++k){
        // 串行部分，也可以尝试并行化
        #pragma omp single
        {
            __m128 diver = _mm_load_ps1(&matrix[k][k]);
		    for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
			    matrix[k][j] = matrix[k][j] / matrix[k][k];
		    for (;j < ROW;j += 4)
		    {
			    __m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			    divee = _mm_div_ps(divee, diver);
			    _mm_storeu_ps(&matrix[k][j], divee);
		    }
		    matrix[k][k] = 1.0;
        }
        // 并行部分，使用行划分
        #pragma omp for
        for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				sub1 =  _mm_loadu_ps(&matrix[i][j]);
				mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver2()//静态分除,不使用SIMD 
{
	int i, j, k; 
	__m128 mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
    for(k = 0; k < ROW; ++k){
        // 并行除法
        #pragma omp for
		for (j = k + 1;j < ROW;++j)
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		matrix[k][k] = 1.0;
        // 并行部分，使用行划分
        #pragma omp for
        for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				sub1 =  _mm_loadu_ps(&matrix[i][j]);
				mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver3()//静态分除,使用SIMD 
{
	int i, j, k, start; 
	__m128 mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1), shared(start)
    for(k = 0; k < ROW; ++k){
        __m128 diver = _mm_load_ps1(&matrix[k][k]);
        //串行处理对齐 
        #pragma omp single
		for (start = k + 1;start < ROW && ((ROW - start) & 3);++start)
			matrix[k][start] = matrix[k][start] / matrix[k][k];
		//并行SIMD
		#pragma omp for 
		for (j = start;j < ROW;j += 4)
		{
			__m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
        // 并行部分，使用行划分
        #pragma omp for
        for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				sub1 =  _mm_loadu_ps(&matrix[i][j]);
				mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver4()//动态任务划分，块大小固定 
{
	int i, j, k; 
	__m128 mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
    for(k = 0; k < ROW; ++k){
        // 串行部分，也可以尝试并行化
        #pragma omp single
        {
            __m128 diver = _mm_load_ps1(&matrix[k][k]);
		    for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
			    matrix[k][j] = matrix[k][j] / matrix[k][k];
		    for (;j < ROW;j += 4)
		    {
			    __m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			    divee = _mm_div_ps(divee, diver);
			    _mm_storeu_ps(&matrix[k][j], divee);
		    }
		    matrix[k][k] = 1.0;
        }
        // 并行部分，使用行划分
        #pragma omp for schedule(dynamic, 15) 
        for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				sub1 =  _mm_loadu_ps(&matrix[i][j]);
				mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver5()//动态任务划分，guided 
{
	int i, j, k; 
	__m128 mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
    for(k = 0; k < ROW; ++k){
        // 串行部分，也可以尝试并行化
        #pragma omp single
        {
            __m128 diver = _mm_load_ps1(&matrix[k][k]);
		    for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
			    matrix[k][j] = matrix[k][j] / matrix[k][k];
		    for (;j < ROW;j += 4)
		    {
			    __m128 divee =  _mm_loadu_ps(&matrix[k][j]);
			    divee = _mm_div_ps(divee, diver);
			    _mm_storeu_ps(&matrix[k][j], divee);
		    }
		    matrix[k][k] = 1.0;
        }
        // 并行部分，使用行划分
        #pragma omp for schedule(guided) 
        for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				sub1 =  _mm_loadu_ps(&matrix[i][j]);
				mult2 =  _mm_loadu_ps(&matrix[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver6()//auto simd
{
	int i, j, k; 
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for(k = 0; k < ROW; ++k){
        //并行化除法 
        #pragma omp for 
		for (j = k + 1;j < ROW;++j)
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		matrix[k][k] = 1.0;
        // 并行部分，使用行划分
        #pragma omp for simd 
        for (i = k + 1; i < ROW; ++i)
		{
			for (int j = i + 1; j < ROW; j++) {
				matrix[k][j] = matrix[k][j] - matrix[i][j] * matrix[k][i];
			}
			matrix[k][i] = 0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver7()//auto simd
{
	int i, j, k; 
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for(k = 0; k < ROW; ++k){
        //并行化除法 
        #pragma omp for simd
		for (j = k + 1;j < ROW;++j)
		{
			matrix[k][j] = matrix[k][j] / matrix[k][k];
			for (i = k + 1; i < ROW; i++) {
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
		}
		#pragma omp single
       	matrix[k][k] = 1.0;
		#pragma omp for simd nowait
		for (i = k + 1; i < ROW; i++)
			matrix[i][k] = 0;
        // 不需要同步 
    }
}

void timing(void (*func)())
{
	ll head, tail, freq;
	double time = 0;
	int counter = 0;
	while (INTERVAL > time)
	{
        init();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
		func();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		counter++;
		time += (tail - head) * 1000.0 / freq;
	}
	std::cout << time / counter << '\n';
}

int main()
{
//	cout<<"串行："; 
//	timing(plain);
//	cout<<"SIMD："; 
//	timing(SIMD);
	for(NUM_THREADS = 5;NUM_THREADS<6;NUM_THREADS++)
	{
//		cout<<"using "<<NUM_THREADS<<" threads"<<endl; 
//	    cout<<"动态限制："; 
//	    timing(ver0);
//	    cout<<"静态朴素："; //1 
//		timing(ver1);
//	    cout<<"朴素分除："; //2 
//		timing(ver2);
//	    cout<<"SIMD分除："; 
//		timing(ver3);
//	    cout<<"dynamic："; //3 
//		timing(ver4);
//	    cout<<"guided："; //4 
//		timing(ver5);
//		cout<<"auto simd"<<endl;
//		timing(ver6); 
//		cout<<"col"<<endl;
//		timing(ver7); 
	}
}
