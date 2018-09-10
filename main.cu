//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <iostream>
// 
//using namespace std;
// 
//int main()
//{
//	cudaDeviceProp deviceProp;
//	int deviceCount;
//	cudaError_t cudaError;
//	cudaError = cudaGetDeviceCount(&deviceCount);
//	for (int i = 0; i < deviceCount; i++)
//	{
//		cudaError = cudaGetDeviceProperties(&deviceProp, i);
// 
//		cout << "设备 " << i + 1 << " 的主要属性： " << endl;
//		cout << "设备显卡型号： " << deviceProp.name << endl;
//		cout << "设备全局内存总量（以MB为单位）： " << deviceProp.totalGlobalMem / 1024 / 1024 << endl;
//		cout << "设备上一个线程块（Block）中可用的最大共享内存（以KB为单位）： " << deviceProp.sharedMemPerBlock / 1024 << endl;
//		cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << deviceProp.regsPerBlock << endl;
//		cout << "设备上一个线程块（Block）可包含的最大线程数量： " << deviceProp.maxThreadsPerBlock << endl;
//		cout << "设备的计算功能集（Compute Capability）的版本号： " << deviceProp.major << "." << deviceProp.minor << endl;
//		cout << "设备上多处理器的数量： " << deviceProp.multiProcessorCount << endl;
//	}
//	getchar();
//	return 0;
//}
/**
 *   addition: C = A + B.
 *
 * This samp le is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */
#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define NX 512
#define NY 512
#define NZ 512

 
using namespace std;
 
void main()
{
	int n = NX;  // 100
	float2 *data_h = new float2[n]; // 创建输入矩阵 注意是float2类型
	for (int i=0; i<n; ++i)
	{
		data_h[i].x = 1;
		data_h[i].y = i;
	}
	float2 *data_d;
	cudaMalloc((void**)&data_d, n * sizeof(float2));  //申请显存空间
	cudaMemcpy(data_d, data_h, n * sizeof(float2), cudaMemcpyHostToDevice);  //将数据拷贝到设备端
 
	static StopWatchInterface *timer;      // 用于测试时间的函数
	sdkCreateTimer(&timer);   // 初始化时间
 
	sdkStartTimer(&timer);  // 开始计时
 
	// 创建CUFFT句柄
	cufftHandle plan1;
	cufftPlan1d(&plan1, n, CUFFT_C2C, 1);
 
	sdkStopTimer(&timer);  // 计时结束
	cout<<"Create plan1: "<<sdkGetTimerValue(&timer)<<"ms"<<endl;
 
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
 
	cufftExecC2C(plan1, data_d, data_d, CUFFT_FORWARD);  // 快速傅里叶正变化
	cufftDestroy(plan1);
 
	sdkStopTimer(&timer);
	cout<<"Execute plan1: "<<sdkGetTimerValue(&timer)<<"ms"<<endl;
 
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
 
	cufftHandle plan2;
	cufftPlan1d(&plan2, n, CUFFT_C2C, 1);
 
	sdkStopTimer(&timer);
	cout<<"Create plan2: "<<sdkGetTimerValue(&timer)<<"ms"<<endl;
 
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
 
	cufftExecC2C(plan2, data_d, data_d, CUFFT_INVERSE);  // 快速傅里叶逆变化
	cufftDestroy(plan2);
 
	sdkStopTimer(&timer);
	cout<<"Execute plan2: "<<sdkGetTimerValue(&timer)<<"ms"<<endl;

	cudaFree(data_d);
	delete data_h;
	

	//---------------------------- 二维傅里叶变换

    n = NX*NY;
	float2 *idata_h = new float2[n]; // 创建输入矩阵 注意是float2类型
	for (int i=0; i<n; ++i)
	{
		idata_h[i].x = i;
		idata_h[i].y = i;
	}
	float2 *idata_d,*odata_d;
	cudaMalloc((void**)&idata_d, n * sizeof(float2));  //申请显存空间
	cudaMalloc((void**)&odata_d, n * sizeof(float2)); 
	cudaMemcpy(idata_d, idata_h, n * sizeof(float2), cudaMemcpyHostToDevice);  //将数据拷贝到设备端
 
	sdkResetTimer(&timer);   // 初始化时间
	sdkStartTimer(&timer);  // 开始计时

	// 创建CUFFT句柄
	cufftHandle plan3;
	cufftPlan2d(&plan3, NX, NY, CUFFT_C2C);

	// 执行CUFFT
	cufftExecC2C(plan3, idata_d, odata_d, CUFFT_FORWARD);  // 快速傅里叶逆变化
		
	sdkStopTimer(&timer);
	cout<<"Execute plan3(2D_FFT): "<<sdkGetTimerValue(&timer)<<"ms"<<endl;

	cufftDestroy(plan3);
	cudaFree(idata_d);
	cudaFree(odata_d);
	delete idata_h;


	//--------------------三维FFT
	n = NX*NY*NZ;
	float2 *idata_h3 = new float2[n]; // 创建输入矩阵 注意是float2类型
	for (int i=0; i<n; ++i)
	{
		idata_h3[i].x = 1;
		idata_h3[i].y = 1;
	}
	float2 *idata_d3,*odata_d3;
	cudaMalloc((void**)&idata_d3, n * sizeof(float2));  //申请显存空间
	cudaMalloc((void**)&odata_d3, n * sizeof(float2)); 
	cudaMemcpy(idata_d3,idata_h3, n * sizeof(float2), cudaMemcpyHostToDevice);  //将数据拷贝到设备端
 
	sdkResetTimer(&timer);   // 初始化时间
	sdkStartTimer(&timer);  // 开始计时

	// 创建CUFFT句柄
	cufftHandle plan4;
	cufftPlan3d(&plan4, NX, NY, NZ, CUFFT_C2C);

	// 执行CUFFT
	cufftExecC2C(plan4, idata_d3, odata_d3, CUFFT_FORWARD);  // 快速傅里叶正变化  512*512*512 用时2.17ms
		
	sdkStopTimer(&timer);
	cout<<"Execute plan4(3D_FFT): "<<sdkGetTimerValue(&timer)<<"ms"<<endl;

	cudaMemcpy(idata_h3, odata_d3,  n * sizeof(float2),cudaMemcpyDeviceToHost);


	cufftDestroy(plan4);
	cudaFree(idata_d3);
	cudaFree(odata_d3);
	delete idata_h3;






	system("pause");
 
}
