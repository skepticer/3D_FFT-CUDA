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
//		cout << "�豸 " << i + 1 << " ����Ҫ���ԣ� " << endl;
//		cout << "�豸�Կ��ͺţ� " << deviceProp.name << endl;
//		cout << "�豸ȫ���ڴ���������MBΪ��λ���� " << deviceProp.totalGlobalMem / 1024 / 1024 << endl;
//		cout << "�豸��һ���߳̿飨Block���п��õ�������ڴ棨��KBΪ��λ���� " << deviceProp.sharedMemPerBlock / 1024 << endl;
//		cout << "�豸��һ���߳̿飨Block���ֿ��õ�32λ�Ĵ��������� " << deviceProp.regsPerBlock << endl;
//		cout << "�豸��һ���߳̿飨Block���ɰ���������߳������� " << deviceProp.maxThreadsPerBlock << endl;
//		cout << "�豸�ļ��㹦�ܼ���Compute Capability���İ汾�ţ� " << deviceProp.major << "." << deviceProp.minor << endl;
//		cout << "�豸�϶ദ������������ " << deviceProp.multiProcessorCount << endl;
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
	float2 *data_h = new float2[n]; // ����������� ע����float2����
	for (int i=0; i<n; ++i)
	{
		data_h[i].x = 1;
		data_h[i].y = i;
	}
	float2 *data_d;
	cudaMalloc((void**)&data_d, n * sizeof(float2));  //�����Դ�ռ�
	cudaMemcpy(data_d, data_h, n * sizeof(float2), cudaMemcpyHostToDevice);  //�����ݿ������豸��
 
	static StopWatchInterface *timer;      // ���ڲ���ʱ��ĺ���
	sdkCreateTimer(&timer);   // ��ʼ��ʱ��
 
	sdkStartTimer(&timer);  // ��ʼ��ʱ
 
	// ����CUFFT���
	cufftHandle plan1;
	cufftPlan1d(&plan1, n, CUFFT_C2C, 1);
 
	sdkStopTimer(&timer);  // ��ʱ����
	cout<<"Create plan1: "<<sdkGetTimerValue(&timer)<<"ms"<<endl;
 
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
 
	cufftExecC2C(plan1, data_d, data_d, CUFFT_FORWARD);  // ���ٸ���Ҷ���仯
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
 
	cufftExecC2C(plan2, data_d, data_d, CUFFT_INVERSE);  // ���ٸ���Ҷ��仯
	cufftDestroy(plan2);
 
	sdkStopTimer(&timer);
	cout<<"Execute plan2: "<<sdkGetTimerValue(&timer)<<"ms"<<endl;

	cudaFree(data_d);
	delete data_h;
	

	//---------------------------- ��ά����Ҷ�任

    n = NX*NY;
	float2 *idata_h = new float2[n]; // ����������� ע����float2����
	for (int i=0; i<n; ++i)
	{
		idata_h[i].x = i;
		idata_h[i].y = i;
	}
	float2 *idata_d,*odata_d;
	cudaMalloc((void**)&idata_d, n * sizeof(float2));  //�����Դ�ռ�
	cudaMalloc((void**)&odata_d, n * sizeof(float2)); 
	cudaMemcpy(idata_d, idata_h, n * sizeof(float2), cudaMemcpyHostToDevice);  //�����ݿ������豸��
 
	sdkResetTimer(&timer);   // ��ʼ��ʱ��
	sdkStartTimer(&timer);  // ��ʼ��ʱ

	// ����CUFFT���
	cufftHandle plan3;
	cufftPlan2d(&plan3, NX, NY, CUFFT_C2C);

	// ִ��CUFFT
	cufftExecC2C(plan3, idata_d, odata_d, CUFFT_FORWARD);  // ���ٸ���Ҷ��仯
		
	sdkStopTimer(&timer);
	cout<<"Execute plan3(2D_FFT): "<<sdkGetTimerValue(&timer)<<"ms"<<endl;

	cufftDestroy(plan3);
	cudaFree(idata_d);
	cudaFree(odata_d);
	delete idata_h;


	//--------------------��άFFT
	n = NX*NY*NZ;
	float2 *idata_h3 = new float2[n]; // ����������� ע����float2����
	for (int i=0; i<n; ++i)
	{
		idata_h3[i].x = 1;
		idata_h3[i].y = 1;
	}
	float2 *idata_d3,*odata_d3;
	cudaMalloc((void**)&idata_d3, n * sizeof(float2));  //�����Դ�ռ�
	cudaMalloc((void**)&odata_d3, n * sizeof(float2)); 
	cudaMemcpy(idata_d3,idata_h3, n * sizeof(float2), cudaMemcpyHostToDevice);  //�����ݿ������豸��
 
	sdkResetTimer(&timer);   // ��ʼ��ʱ��
	sdkStartTimer(&timer);  // ��ʼ��ʱ

	// ����CUFFT���
	cufftHandle plan4;
	cufftPlan3d(&plan4, NX, NY, NZ, CUFFT_C2C);

	// ִ��CUFFT
	cufftExecC2C(plan4, idata_d3, odata_d3, CUFFT_FORWARD);  // ���ٸ���Ҷ���仯  512*512*512 ��ʱ2.17ms
		
	sdkStopTimer(&timer);
	cout<<"Execute plan4(3D_FFT): "<<sdkGetTimerValue(&timer)<<"ms"<<endl;

	cudaMemcpy(idata_h3, odata_d3,  n * sizeof(float2),cudaMemcpyDeviceToHost);


	cufftDestroy(plan4);
	cudaFree(idata_d3);
	cudaFree(odata_d3);
	delete idata_h3;






	system("pause");
 
}
