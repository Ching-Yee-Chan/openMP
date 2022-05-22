#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <arm_neon.h>
#include <stdio.h>
#include <sys/time.h>
#define ROW 2048
#define TASK 8
#define INTERVAL 10000
using namespace std;
float matrix[ROW][ROW];
float revmat[ROW][ROW];
typedef long long ll;
//静态线程数量
int NUM_THREADS = 4;

void reverse()
{
	for (int i = 0;i < ROW;i++)
		for (int j = 0;j < ROW;j++)
			revmat[j][i] = matrix[i][j];
}

void init()
{
	for (int i = 0;i < ROW;i++)
	{
		for (int j = 0;j < i;j++)
			matrix[i][j] = 0;
		for(int j = i;j<ROW;j++)
			matrix[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0;k < 1000;k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0;j < ROW;j++)
			matrix[row1][j] += mult * matrix[row2][j];
	}
	reverse();
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
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)
		{//消去
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			int j;
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
	}
}

void ColSIMD()
{
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)//i为列
		{//逐列消去
			int j;//j为目标行
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_lane_f32(&matrix[j][i], subbee, 0);
				mult1 = vld1q_lane_f32(&matrix[j][k], mult1, 0);
				subbee = vld1q_lane_f32(&matrix[j + 1][i], subbee, 1);
				mult1 = vld1q_lane_f32(&matrix[j + 1][k], mult1, 1);
				subbee = vld1q_lane_f32(&matrix[j + 2][i], subbee, 2);
				mult1 = vld1q_lane_f32(&matrix[j + 2][k], mult1, 2);
				subbee = vld1q_lane_f32(&matrix[j + 3][i], subbee, 3);
				mult1 = vld1q_lane_f32(&matrix[j + 3][k], mult1, 3);
				mult2 = vld1q_dup_f32(&matrix[k][i]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_lane_f32(&matrix[j][i], subbee, 0);
				vst1q_lane_f32(&matrix[j + 1][i], subbee, 1);
				vst1q_lane_f32(&matrix[j + 2][i], subbee, 2);
				vst1q_lane_f32(&matrix[j + 3][i], subbee, 3);
			}
		}
		for (int i = k + 1; i < ROW; i++)
			matrix[i][k] = 0;
	}
}

void ColSIMDcached()
{
	for (int k = 0;k < ROW;++k)
	{
		float32x4_t diver = vld1q_dup_f32(&revmat[k][k]);
		int j;
		for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
			revmat[j][k] = revmat[j][k] / revmat[k][k];
		for (;j < ROW;j += 4)
		{
			float32x4_t divee = vld1q_lane_f32(&revmat[j][k], divee, 0);
			divee = vld1q_lane_f32(&revmat[j+1][k], divee, 1);
			divee = vld1q_lane_f32(&revmat[j+2][k], divee, 2);
			divee = vld1q_lane_f32(&revmat[j+3][k], divee, 3);
			divee = vdivq_f32(divee, diver);
			vst1q_lane_f32(&revmat[j][k], divee, 0);
			vst1q_lane_f32(&revmat[j + 1][k], divee, 1);
			vst1q_lane_f32(&revmat[j + 2][k], divee, 2);
			vst1q_lane_f32(&revmat[j + 3][k], divee, 3);
		}
		revmat[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)//i为行
		{//轉置前逐列消去，轉置后逐行消去
			int j;//j为目标列
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_f32(&revmat[i][j]);
				mult1 = vld1q_f32(&revmat[k][j]);
				mult2 = vld1q_dup_f32(&revmat[i][k]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_f32(&revmat[i][j], subbee);
			}
		}
		for (int i = k + 1; i < ROW; i++)
			revmat[k][i] = 0;
	}
}

void ver0()//动态线程 
{
	int i, j, k; 
	float32x4_t mult1, mult2, sub1;
	for (int k = 0; k < ROW; ++k)
	{
        float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
	    for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
		    matrix[k][j] = matrix[k][j] / matrix[k][k];
		for (;j < ROW;j += 4)
	    {
			float32x4_t divee = vld1q_f32(&matrix[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&matrix[k][j], divee);
	    }
		matrix[k][k] = 1.0;
		#pragma omp parallel for num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
        for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void ver1()//静态线程 
{
	int i, j, k; 
	float32x4_t mult1, mult2, sub1;
	#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (int k = 0; k < ROW; ++k)
	{
		#pragma omp single
        {
            float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		    for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
			    matrix[k][j] = matrix[k][j] / matrix[k][k];
		    for (;j < ROW;j += 4)
		    {
				float32x4_t divee = vld1q_f32(&matrix[k][j]);
				divee = vdivq_f32(divee, diver);
				vst1q_f32(&matrix[k][j], divee);
		    }
		    matrix[k][k] = 1.0;
        }
		#pragma omp for
        for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void ver2()//静态分除,不使用SIMD 
{
	int i, j, k; 
	float32x4_t mult1, mult2, sub1;
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
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver3()//静态分除,使用SIMD 
{
	int i, j, k, start; 
	float32x4_t mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1), shared(start)
    for(k = 0; k < ROW; ++k){
        float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
        //串行处理对齐 
        #pragma omp single
		for (start = k + 1;start < ROW && ((ROW - start) & 3);++start)
			matrix[k][start] = matrix[k][start] / matrix[k][k];
		//并行SIMD
		#pragma omp for 
		for (j = start;j < ROW;j += 4)
		{
				float32x4_t divee = vld1q_f32(&matrix[k][j]);
				divee = vdivq_f32(divee, diver);
				vst1q_f32(&matrix[k][j], divee);
		}
		matrix[k][k] = 1.0;
        // 并行部分，使用行划分
        #pragma omp for
        for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver4()//列划分，cache优化前 
{
	int i, j, k; 
	float32x4_t subbee, mult1, mult2;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, subbee, mult1, mult2)
    for(k = 0; k < ROW; ++k){ 
        #pragma omp for
		for (j = k + 1;j < ROW;++j)
		{
			//除法 
			matrix[k][j] = matrix[k][j] / matrix[k][k];
			//消去 
			for (int i = k + 1; i < ROW ; i++)//i为列
			{//逐列消去
				for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
					matrix[j][i] = matrix[j][i] - matrix[j][k] * matrix[k][i];
				for (;j < ROW;j += 4)
				{
					subbee = vld1q_lane_f32(&matrix[j][i], subbee, 0);
					mult1 = vld1q_lane_f32(&matrix[j][k], mult1, 0);
					subbee = vld1q_lane_f32(&matrix[j + 1][i], subbee, 1);
					mult1 = vld1q_lane_f32(&matrix[j + 1][k], mult1, 1);
					subbee = vld1q_lane_f32(&matrix[j + 2][i], subbee, 2);
					mult1 = vld1q_lane_f32(&matrix[j + 2][k], mult1, 2);
					subbee = vld1q_lane_f32(&matrix[j + 3][i], subbee, 3);
					mult1 = vld1q_lane_f32(&matrix[j + 3][k], mult1, 3);
					mult2 = vld1q_dup_f32(&matrix[k][i]);
					mult1 = vmulq_f32(mult1, mult2);
					subbee = vsubq_f32(subbee, mult1);
					vst1q_lane_f32(&matrix[j][i], subbee, 0);
					vst1q_lane_f32(&matrix[j + 1][i], subbee, 1);
					vst1q_lane_f32(&matrix[j + 2][i], subbee, 2);
					vst1q_lane_f32(&matrix[j + 3][i], subbee, 3);
				}
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

void ver5()//列划分，cache优化 
{
	int i, j, k; 
	float32x4_t subbee, mult1, mult2;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, subbee, mult1, mult2)
	for (k = 0;k < ROW;++k)
	{
		#pragma omp for
		for (i = k + 1; i < ROW; i++)//i为列
		{//逐列消去
			revmat[i][k] = revmat[i][k] / revmat[k][k];
			for (j = k + 1;j < ROW && ((ROW - j) & 3);j++)//处理串行部分
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
			for (;j < ROW;j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_f32(&revmat[i][j]);
				mult1 = vld1q_f32(&revmat[k][j]);
				mult2 = vld1q_dup_f32(&revmat[i][k]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_f32(&revmat[i][j], subbee);
			}
		}
		#pragma omp single
       	revmat[k][k] = 1.0;
		#pragma omp for simd nowait
		for (int j = k + 1;j < ROW;j++)
			revmat[k][j] = 0;
		//处理已被消去的列
	}
}

void ver6()//auto simd horizontal
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

void ver7()//auto simd col
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

void ver10()//auto simd col cached
{
	int i, j, k; 
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
    for (k = 0;k < ROW;++k)
	{
		#pragma omp for
		for (i = k + 1; i < ROW; i++)//i为列
		{//逐列消去
			revmat[i][k] = revmat[i][k] / revmat[k][k];
			for (j = k + 1;j < ROW;j++)//处理串行部分
				revmat[i][j] = revmat[i][j] - revmat[k][j] * revmat[i][k];
		}
		#pragma omp single
       	revmat[k][k] = 1.0;
		#pragma omp for simd nowait
		for (int j = k + 1;j < ROW;j++)
			revmat[k][j] = 0;
		//处理已被消去的列
	}
}

void ver8()//动态任务划分，块大小固定 
{
	int i, j, k; 
	float32x4_t mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
    for(k = 0; k < ROW; ++k){
        // 串行部分，也可以尝试并行化
        #pragma omp single
        {
            float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		    for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
			    matrix[k][j] = matrix[k][j] / matrix[k][k];
		    for (;j < ROW;j += 4)
		    {
				float32x4_t divee = vld1q_f32(&matrix[k][j]);
				divee = vdivq_f32(divee, diver);
				vst1q_f32(&matrix[k][j], divee);
		    }
		    matrix[k][k] = 1.0;
        }
        // 并行部分，使用行划分
        #pragma omp for schedule(dynamic, 7) 
        for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void ver9()//动态任务划分，guided 
{
	int i, j, k; 
	float32x4_t mult1, mult2, sub1;
    // 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
    //删除下面一句即为动态线程 
    #pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
    for(k = 0; k < ROW; ++k){
        // 串行部分，也可以尝试并行化
        #pragma omp single
        {
            float32x4_t diver = vld1q_dup_f32(&matrix[k][k]);
		    for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)
			    matrix[k][j] = matrix[k][j] / matrix[k][k];
		    for (;j < ROW;j += 4)
		    {
				float32x4_t divee = vld1q_f32(&matrix[k][j]);
				divee = vdivq_f32(divee, diver);
				vst1q_f32(&matrix[k][j], divee);
		    }
		    matrix[k][k] = 1.0;
        }
        // 并行部分，使用行划分
        #pragma omp for schedule(guided) 
        for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&matrix[i][k]);
			for (j = k + 1;j < ROW && ((ROW - j) & 3);++j)//串行处理对齐
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			for (;j < ROW;j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&matrix[i][j]);
				float32x4_t mult2 = vld1q_f32(&matrix[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&matrix[i][j], sub1);
			}
			matrix[i][k] = 0.0;
		}
        // 离开for循环时，各个线程默认同步，进入下一行的处理
    }
}

void timing(void (*func)())
{
	timeval tv_begin, tv_end;
	int counter(0);
	double time = 0;
	while (INTERVAL > time)
	{
		init();
		gettimeofday(&tv_begin, 0);
		func();
		gettimeofday(&tv_end, 0);
		counter++;
		time += ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec) * 1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec) / 1000.0;
	}
	cout << time / counter << "," << counter << '\n';
}

int main()
{
	cout<<"串行："; 
	timing(plain);
	cout<<"SIMD："; 
	timing(SIMD);
	// cout<<"ColSIMD："; 
	// timing(ColSIMD);
	// cout<<"ColSIMDcached："; 
	// timing(ColSIMDcached);
	for(NUM_THREADS = 8;NUM_THREADS<=8;NUM_THREADS++)
	{
		// cout<<"using "<<NUM_THREADS<<" threads"<<endl; 
	    // cout<<"动态限制："; 
	    // timing(ver0);
	    // cout<<"静态朴素："; 
		//timing(ver1);
	    cout<<"朴素分除："; 
		timing(ver2);
	    cout<<"SIMD分除："; 
		timing(ver3);
	    // cout<<"按列："; 
		// timing(ver4);
	    // cout<<"按列cached："; 
		// timing(ver5);
		cout<<"auto simd horizontal：";
		timing(ver6); 
		cout<<"auto simd vertical：";
        timing(ver10); 
		// cout<<"auto simd vertical cached：";
		// timing(ver7); 
		cout<<"dynamic：";
		timing(ver8); 
		cout<<"guided：";
		timing(ver9); 
	}
}
