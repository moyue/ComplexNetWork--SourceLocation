
#include "cuda_runtime.h"
//#include "common_functions.h"
//#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <dos.h>
#include "GuassRandom.h"
#include "graph.h"
#include "SqQueue.h"
#define BLOCK_DIM 16  //COUNT-1的值最好为BLOCK_DIM 的整数倍  
//4时最多是13个观察点       4*4个线程块
//8时最多是25个观察点 即最大24*24的矩阵  （硬件限制？512个线程）最大3*3个线程块         
//9  2*2 19
//10 2*2 21
//16 2*2 33
//20 1*1 21
#define ARRIVAL_TIME 199
#include "device_functions.h"


int *activeObservers;
int COUNT = -1;

cudaError_t MatrixWithCuda(float *csFinal,float *csEnd, int height,int width);
//
//__global__ void cuda_Matrix(float* odata, float* idata, int width, int height)
//{
//	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
//	// read matrix tile into shared memory
//	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//	if ((xIndex < width ) && (yIndex < height))
//	{
//		unsigned int index_in = yIndex * width + xIndex;
//		block[threadIdx.y][threadIdx.x] = idata[index_in];
//	}
//
//	__syncthreads();
//	// write the transposed matrix tile to global memory
//	xIndex = blockIdx.y * blockDim.y + threadIdx.x;
//	yIndex = blockIdx.x * blockDim.x + threadIdx.y;
//	if((xIndex < height) && (yIndex < width))
//	{
//		unsigned int index_out = yIndex * height + xIndex;
//		odata[index_out] = block[threadIdx.x][threadIdx.y];
//	}
//}
// 矩阵求逆之相消操作
__global__ void MatrixInverse_Elimination(float *A,float *B,int n,int i)
{
	//共享存储矩阵块
	__shared__ float As[BLOCK_DIM][BLOCK_DIM];
	__shared__ float Bs[BLOCK_DIM][BLOCK_DIM];
	int tix = blockDim.x * blockIdx.x + threadIdx.x;
	int tiy = blockDim.y * blockIdx.y + threadIdx.y;

	if(tiy<n && tix<n)
	{
		//拷贝block到共享存储区
		As[threadIdx.y][threadIdx.x] = A[tiy*n+tix];
		Bs[threadIdx.y][threadIdx.x] = B[tiy*n+tix];
		__syncthreads();

			float a = A[tiy*n+i]*A[i*n+tix];
			float b = A[tiy*n+i]*B[i*n+tix];
			
			//用归一化的第i行与非i行相消
			As[threadIdx.y][threadIdx.x] = As[threadIdx.y][threadIdx.x] - a;
			Bs[threadIdx.y][threadIdx.x] = Bs[threadIdx.y][threadIdx.x] - b;

		__syncthreads();
		if(tiy != i)        //将相消结果存入显存
		{
			A[tiy*n+tix] = As[threadIdx.y][threadIdx.x];
			B[tiy*n+tix] = Bs[threadIdx.y][threadIdx.x];
		}
		//__syncthreads();
	}
}
__global__ void MatrixInverse_Normalized(float *A,float *B,int n,int i)
{
	int tix = blockDim.x * blockIdx.x + threadIdx.x;
	int tiy = blockDim.y * blockIdx.y + threadIdx.y;
	float temp;//归一化值
	temp = A[i*n+i];
	//对A,B矩阵第i行做归一化操作
	if(tiy<n && tix<n)
	{
		if (tix == i)
		{
		
			A[tix*n + tiy] /=temp;
			B[tix*n + tiy] /=temp;
		}
		//__syncthreads();
	}
}


void randomSetWeight(ALGraph *graph)//可以讲此步骤放到生成图的过程中
{
	float a[250] = {11.0,10.0,17.0,14.0,7.0,9.0,8.0,12.0,13.0,8.0,20.0,7.0,12.0,9.0,14.0,8.0,12.0,12.0,13.0,16.0,7.0,13.0,10.0,9.0,12.0,12.0,15.0,10.0,13.0,12.0,14.0,17.0,14.0,8.0,12.0,12.0,12.0,7.0,10.0,12.0,12.0,12.0,9.0,10.0,12.0,10.0,5.0,14.0,10.0,10.0,6.0,10.0,11.0,10.0,16.0,11.0,14.0,15.0,12.0,12.0,13.0,12.0,10.0,12.0,13.0,11.0,7.0,8.0,7.0,10.0,9.0,13.0,10.0,15.0,11.0,15.0,12.0,12.0,11.0,10.0,14.0,14.0,10.0,14.0,11.0,6.0,14.0,10.0,9.0,12.0,13.0,9.0,12.0,14.0,12.0,11.0,13.0,12.0,13.0,9.0,20.0,7.0,15.0,11.0,11.0,14.0,13.0,14.0,12.0,10.0,12.0,19.0,11.0,14.0,8.0,9.0,11.0,8.0,10.0,9.0,7.0,10.0,14.0,7.0,13.0,13.0,4.0,8.0,5.0,10.0,5.0,8.0,8.0,7.0,13.0,15.0,15.0,14.0,11.0,12.0,12.0,7.0,7.0,8.0,8.0,10.0,6.0,8.0,8.0,15.0,15.0,12.0,12.0,11.0,8.0,10.0,7.0,15.0,13.0,7.0,16.0,7.0,14.0,12.0,18.0,11.0,9.0,11.0,8.0,14.0,15.0,13.0,10.0,12.0,11.0,9.0,13.0,10.0,9.0,7.0,12.0,8.0,13.0,11.0,10.0,13.0,11.0,13.0,11.0,12.0,10.0,11.0,16.0,13.0,7.0,11.0,10.0,14.0,10.0,15.0,5.0,15.0,11.0,6.0,14.0,9.0,11.0,12.0,11.0,9.0,16.0,17.0,11.0,10.0,13.0,6.0,8.0,15.0,12.0,13.0,16.0,14.0,15.0,6.0,10.0,11.0,7.0,12.0,10.0,12.0,11.0,15.0,9.0,17.0,6.0,7.0,10.0,16.0,10.0,8.0,9.0,9.0,9.0,12.0,13.0,9.0,17.0,12.0,7.0,13.0};

	for(int i =0; i< graph->arcnum; i++)
	{
		//graph->arctices[i].Weight = (int)(3*GetOneGaussian(0,1.0)+12);
		//graph->arctices[i].tmpWeight = graph->arctices[i].Weight;
		 		graph->arctices[i].Weight = a[i];
		 		graph->arctices[i].tmpWeight = a[i];
		//	printf("%d+%d\n" ,i,graph->arctices[i].Weight);
	}
}
void setAttribute(ALGraph *graph)
{
	for (int i =0 ;i<graph->vexnum;i++)
	{
		graph->vertices[i].time = -1;
		graph->vertices[i].isActive = false;
		graph->vertices[i].direction = -1;
	}
}

//int diffArray[DIFF_EDGE_NUM];
int *diffArray;
static int diffLength = 0;
void help_add(int arcNode)
{
	diffArray[diffLength] = arcNode;
	diffLength ++;
}
void help_remove(int temp)
{
	for(int j=0;j<diffLength;j++)
		if(diffArray[j]==temp)
		{
			diffArray[j]=-1;
			break;
		}     
}
bool help_contain(int temp)
{

	for(int j=0;j<diffLength;j++)
	{
		if(diffArray[j]==temp)
			return true;
	}
	return false;
}
/************************************************************************/
/* 消息传播                                                                     */
/************************************************************************/
void diffusion(ALGraph *graph, int source)//source为realNode的节点ID
{
	int time = ARRIVAL_TIME;//199
	int zerotime = 1;
	int sourceNodeIndex = GetNodeIndex(*graph,source);
	int sourceDegree = graph->vertices[sourceNodeIndex].degree;
	//	int *edges;
	//	edges = (int *)malloc(sourceDegree * sizeof(int));
	//int edges[sourceDegree]; 
	int *edges = getNodeEdges(graph,source);
	int i;
	//     for (i =0 ;i<sourceDegree;i++)
	//     {
	// 		printf("%d ",edges[i]);//源节点的临边:边的ID
	//     }
	diffLength = 0;
	if (diffArray != NULL)
	{
		free(diffArray);
	}
	diffArray = (int *)malloc((graph->arcnum)*sizeof(int));
	for (i=0;i<sourceDegree;i++)
	{
		help_add(edges[i]);
	}

	graph->vertices[sourceNodeIndex].time=1;
	graph->vertices[sourceNodeIndex].direction=-1;
	graph->vertices[sourceNodeIndex].isActive=true;

	while(zerotime<time)
	{
		for (i=0;i<diffLength;i++)
		{
			if (diffArray[i]!=-1)
			{
				//diffArray[i]中的值为边的ID，也是边在graph中的数组索引
				//edgeID = diffArray[i] = arc.arcid
				Arc  *arc= &(graph->arctices[diffArray[i]]);
				int weight = arc->tmpWeight;
				if (weight-1==0)
				{
					int s = arc->arcsourceId;
					int t = arc->arctargerId;
					if (!(graph->vertices[GetNodeIndex(*graph,t)].isActive))
					{
						graph->vertices[GetNodeIndex(*graph,t)].isActive = true;
						graph->vertices[GetNodeIndex(*graph,t)].direction = s;
						graph->vertices[GetNodeIndex(*graph,t)].time = zerotime+1;
						int *tids = getNodeEdges(graph,t);
						int tdegree = graph->vertices[GetNodeIndex(*graph,t)].degree;
						for (int j=0;j<tdegree;j++)
						{
							Arc *arct =&(graph->arctices[tids[j]]);
							int tnode = arct->arcsourceId;
							if (tnode == t)
							{
								tnode = arct->arctargerId;
							} 
							if (!help_contain(arct->arcId) && !(graph->vertices[GetNodeIndex(*graph,tnode)].isActive))
							{
								help_add(arct->arcId);
								arct->tmpWeight = arct->tmpWeight + 1;
							}
						}
						help_remove(arc->arcId);
					} 
					else if(!(graph->vertices[GetNodeIndex(*graph,s)].isActive))
					{
						graph->vertices[GetNodeIndex(*graph,s)].isActive = true;
						graph->vertices[GetNodeIndex(*graph,s)].direction = t;
						graph->vertices[GetNodeIndex(*graph,s)].time = zerotime+1;
						int *sids = getNodeEdges(graph,s);
						int sdegree = graph->vertices[GetNodeIndex(*graph,s)].degree;
						for (int j=0;j<sdegree;j++)
						{
							Arc *arcs = &(graph->arctices[sids[j]]);
							int snode = arcs->arcsourceId;
							if (snode == s)
							{
								snode = arcs->arctargerId;
							} 
							if (!help_contain(arcs->arcId) && !(graph->vertices[GetNodeIndex(*graph,snode)].isActive))
							{
								help_add(arcs->arcId);
								arcs->tmpWeight = arcs->tmpWeight + 1;
							}
						}
						help_remove(arc->arcId);
					}else
					{
						help_remove(arc->arcId);
					}
				}
				arc->tmpWeight = arc->tmpWeight -1;
			}
		}
		zerotime++;
	}
	free(edges);
}
bool isarray_contain(int *tempArray,int temp)//用于生产随机数的去重
{

	for(int j=0;j<COUNT;j++)
	{
		if(tempArray[j]==temp)
			return true;//存在temp返回true
	}
	return false;
}
/************************************************************************/
/* 随机选择观察点策略                 */
/************************************************************************/
void selectObserversAsRandom(ALGraph graph)
{
	if (activeObservers != NULL)
	{
		free(activeObservers);
	}
	activeObservers = (int *)malloc(COUNT*sizeof(int));
	int *temp = (int *)malloc(COUNT*sizeof(int));
	int i = 0;
	while(i<COUNT)
	{
		int randIndex = rand()%graph.vexnum;
		if (!isarray_contain(temp,randIndex)&&graph.vertices[randIndex].isActive)
		{
			temp[i] = randIndex;
			//	printf("temp[%d]=%d \n",i,temp[i]);
			//	activeObservers[i] = i+1;
			activeObservers[i] = graph.vertices[randIndex].nodeid;
			i++;
			//printf("nodeId %d\n",activeObservers[i]);
		}
	}
	// 	for (i =0;i<COUNT;i++)
	// 	{
	// 		//printf("%d\n",activeObservers[i]);
	// 	}
	free(temp);
}
/************************************************************************/
/* 按照度排列策略选择观察点  */
/* 按最大度排列选择，按最小度选择*/ //测试效果按度小的选择比较好
/************************************************************************/
void selectObserversAsDegree(ALGraph graph)
{
	if (activeObservers != NULL)
	{
		free(activeObservers);
	}
	activeObservers = (int *)malloc(COUNT*sizeof(int));
	int *temp = (int *)malloc(COUNT*sizeof(int));
	int maxDegree = 0; 
	//int minDegree = 1000;
	int maxDegreeNodeID = 0;
	//for (int i=0; i<COUNT;i++)
	int i = 0;
//	int num =0;
	while(i<COUNT)
	{

		for (int j=0; j<graph.vexnum; j++)
		{
			//if (!isarray_contain(temp,graph.vertices[j].nodeid) && graph.vertices[j].degree<maxDegree)
			if (!isarray_contain(temp,graph.vertices[j].nodeid) && graph.vertices[j].degree>maxDegree)
			{

				maxDegreeNodeID = graph.vertices[j].nodeid;
				maxDegree = graph.vertices[j].degree;
				//minDegree = graph.vertices[j].degree;
			}
		}	
		if (graph.vertices[GetNodeIndex(graph,maxDegreeNodeID)].isActive)
		{
			temp[i] = maxDegreeNodeID;
			activeObservers[i] = maxDegreeNodeID;
			i++;
		}	
		maxDegree = 0;
		//	minDegree = 1000;
		maxDegreeNodeID = 0;
	}
	printf("观察点为：\n");
	for (i =0;i<COUNT;i++)
	{
		printf("%d\n",activeObservers[i]);
	}
	free(temp);
}
void setObserver(ALGraph *graph)
{
	for (int i=0; i<graph->arcnum; i++)
	{
		graph->arctices[i].tmpWeight = graph->arctices[i].Weight;
	}

}
void propagation(ALGraph *graph,int sourceid)
{
	setAttribute(graph);
	diffusion(graph,sourceid);
	//	selectObserversAsRandom(*graph);
	selectObserversAsDegree(*graph);
	setObserver(graph);// 主要目的是将tmpweight还原，供下一次循环使用
}
bool isset_contain(int *tempArray,int temp,int arraylenth)//
{

	for(int j=0;j<arraylenth;j++)
	{
		if(tempArray[j]==temp)
			return true;
	}
	return false;
}
void generatorBFSTress(ALGraph *graph,int rootId)
{
	SqQueue Q;
	InitQueue(Q);
	int *set;
	// 	if (set !=NULL )
	// 	{
	// 		free(set);
	// 	}
	set = (int *)malloc((graph->vexnum)*sizeof(int));
	for (int i=0 ;i<graph->vexnum;i++)
	{
		set[i] = -1;
	}
	int  len;
	graph->vertices[GetNodeIndex(*graph,rootId)].parent = -1;
	graph->vertices[GetNodeIndex(*graph,rootId)].plength = 0;
	set[0]  = rootId;
	int setI = 1;
	EnQueue(Q,rootId);
	while(!QueueEmpty(Q))
	{
		int qNodeId ;
		DeQueue(Q,qNodeId);
		len = graph->vertices[GetNodeIndex(*graph,qNodeId)].plength;

		int *neighbors = getNodeEdges(graph,qNodeId);//分析对每个相邻的边，亦即遍历与这个节点相临的节点。返回边的ID
		for (int n =0; n<graph->vertices[GetNodeIndex(*graph,qNodeId)].degree; n++)
		{
			int nId = -1; // 得到一个相邻的节点ID
			if (graph->arctices[neighbors[n]].arcsourceId==qNodeId)
			{
				nId = graph->arctices[neighbors[n]].arctargerId;
			} 
			else if (graph->arctices[neighbors[n]].arctargerId==qNodeId)
			{
				nId = graph->arctices[neighbors[n]].arcsourceId;
			}
			if (!isset_contain(set,nId,graph->vexnum))
			{
				set[setI] = nId;
				setI++;
				EnQueue(Q,nId);
				graph->vertices[GetNodeIndex(*graph,nId)].parent = qNodeId;
				graph->vertices[GetNodeIndex(*graph,nId)].plength = len +1;

			}

		}
	}
	free(set);
	DestroyQueue(Q);
}
void cMatrix(float *a, int n, float *end)
{
	//	float Vp[][] = new float[n + 1][2 * n + 1];
	float **Vp;
	int xi,yj;
	Vp = (float **)malloc(sizeof(float *)*(n+1));
	for (int vpi=0;vpi<n+1;vpi++)
	{
		Vp[vpi] = (float *) malloc((sizeof(float)*(2*n+1)));
	}
	for (xi=0; xi<n+1; xi++)
	{
		for (yj=0; yj<2*n+1; yj++)
		{
			Vp[xi][yj] =0.0;//要对初始化矩阵赋初值，否则C的初值不为0
		}
	}
	int i, j,k;
	i = j = n;
	// 	for ( k = 1; k < n + 1; k++) {
	// 		for (int t = 1; t < 2 * n + 1; t++) {
	// 			Vp[k][t] = 0.0;
	// 		}
	// 	}
	//input the data of the matrix
	for (k = 1; k <= n; k++) {
		for (int t = 1; t <= n; t++) {
			Vp[k][t] = a[(k - 1)*n+t - 1];
		}
	}


	for (k = 1; k <= i; k++) {
		for (int t = j + 1; t <= j * 2; t++) {
			if ((t - k) == j) {
				Vp[k][t] = 1.0;
			} else {
				Vp[k][t] = 0;
			}
		}
	}

	for (k = 1; k <= i; k++) {
		if (Vp[k][k] != 1) {
			float bs = Vp[k][k];
			Vp[k][k] = 1;
			for (int p = k + 1; p <= j * 2; p++) {
				Vp[k][p] /= bs;
			}
		}

		for (int q = 1; q <= i; q++) {
			if (q != k) {
				float bs = Vp[q][k];
				for (int p = 1; p <= j * 2; p++) {
					Vp[q][p] -= bs * Vp[k][p];
				}
			} else {
				continue;
			}
		}
	}
	//print out the result of the change
	//  System.out.println("---------------\nThe result is:");
	for (int x = 1; x <= i; x++) {
		for (int y = j + 1; y <= j * 2; y++) {
			//a[x-1][y-2]=Vp[x][y];


			//end[x - 1][y - j - 1] = Vp[x][y];
			end[(x-1)*n+y-j-1] = Vp[x][y];

			//System.out.print(Vp[x][y] + "  ");
		}
		//System.out.println();
	}
}
int main()
{
	printf("**************请输入观察点的个数*****************\n");
	printf("COUNT>");
	scanf("%d",&COUNT);
	if (COUNT <=0)
	{
		printf("COUNT<=0");
		exit(0);
	}
	ALGraph graph;
	CreateGraph(&graph);
	Display(graph);
	srand((unsigned)time(NULL));//产生随机数应用
	randomSetWeight(&graph);

	int hit = 0;
	//第一层for对所有节点，做测试
	for (int i =0; i<graph.vexnum;i++)//孤立节点情况，输入图要为最大连通子图
	{

		int realSourceNodeId = graph.vertices[i].nodeid;
		propagation(&graph,realSourceNodeId);

		//估计出当前节点为源点时，设置传播过程，选择一种部署策略，求出估计的最大值的节点ID，并测量估计值与源点的位置差距，计算误差率
		float maxEstimator = -100000;
		int maxNodeId = -1;
		//	printf("\n********延迟向量的值***********\n");
		int activeObserversSize = COUNT;
		int *delay = (int*)malloc((activeObserversSize-1)*sizeof(int));//activeObseversize 为COUNT-1
		int refObseverIndex = GetNodeIndex(graph,activeObservers[0]);
		int reftime = graph.vertices[refObseverIndex].time;

		for (int delayi=1; delayi<COUNT; delayi++)
		{	
			int curObseverIndex = GetNodeIndex(graph,activeObservers[delayi]);
			delay[delayi-1] = graph.vertices[curObseverIndex].time - reftime;
			// printf("%d\n",delay[delayi-1]);	//延迟向量d的值
		}

		for (int exti=0; exti<graph.vexnum;exti++)
		{
			//printf("BFS Start\n");
			generatorBFSTress(&graph,graph.vertices[exti].nodeid);//生成BFS树
			//printf("BFS End\n");
			float *us = (float *)malloc((activeObserversSize-1)*sizeof(float));
			float *usT = (float *)malloc((activeObserversSize-1)*sizeof(float));
			float refus = graph.vertices[refObseverIndex].plength;
			//printf("us Start");
			//	printf("%f\n",us[0]);
			for (int oi = 1; oi<COUNT; oi++)
			{
				int curObseverIndex = GetNodeIndex(graph,activeObservers[oi]);//activeObservers为节点ID值
				//	printf("%f\n",graph.vertices[curObseverIndex].plength);
				us[oi-1] = (graph.vertices[curObseverIndex].plength - refus)*0.5;
				usT[oi - 1] = us[oi - 1] * 2 * 12;
				//	printf("%f\n",us[oi-1]);

			}
			for (int j=0 ; j<activeObserversSize-1;j++)
			{
				us[j] = delay[j] - us[j]*12;//d-0.5*us
			}
		//	printf("us end");
			int **cs;
			cs = (int **)malloc(sizeof(int *)*activeObserversSize);
			//cs[0] = (int *) malloc((sizeof(int)*activeObserversSize*graph.arcnum));
			for (int csi=0;csi<activeObserversSize;csi++)
			{
				cs[csi] =  (int *) malloc((sizeof(int)*graph.arcnum));
			}
			int x,y;
			for (x=0; x<activeObserversSize; x++)
			{
				for (y=0; y<graph.arcnum; y++)
				{
					cs[x][y] =0;//要对初始化矩阵赋初值，否则C的初值不为0
				}
			}
			for (int k=0; k<activeObserversSize; k++)//计算CS，，得每个观察点到源点的路径
			{
				int observerNodeID = activeObservers[k];
				while(graph.vertices[GetNodeIndex(graph,observerNodeID)].parent != -1)
				{
					int parentid = graph.vertices[GetNodeIndex(graph,observerNodeID)].parent;
					int edgeid = GetEdgeId(graph,parentid,observerNodeID);
					cs[k][edgeid] = 1;
					observerNodeID = parentid;
				}
			}
			float *csFinal;
			csFinal = (float *)malloc(sizeof(float)*(activeObserversSize-1)*(activeObserversSize-1));
			for (x=0; x<(activeObserversSize-1)*(activeObserversSize-1); x++)
			{
				/*for (y=0; y<activeObserversSize-1; y++)
				{*/
					csFinal[x] =0.0;//要对初始化矩阵赋初值，否则C的初值不为0
				//}
			}
			float *csEnd;
			csEnd = (float *)malloc(sizeof(float)*(activeObserversSize-1)*(activeObserversSize-1));

		
			//将矩阵的每一行减去第一行
			int m,n;//循环变量
			for(m=1; m<activeObserversSize;m++)
			{
				n=0;
				while(n < graph.arcnum)
				{
					cs[m][n] = cs[m][n] - cs[0][n];
					//	printf("%d ",cs[m][n]);
					n++;
				}
			}
			int sum = 0;
			for (m=1; m<activeObserversSize;m++)
			{
				for (n=1; n<activeObserversSize; n++)
				{
					int l = 0;
					while(l < graph.arcnum)
					{
						sum = sum + cs[n][l] * cs[m][l];
						l++;
					}
					//csFinal[m-1][n-1] = sum *9;
					csFinal[(m-1)*(activeObserversSize-1)+n-1] = (float)sum * 9;
					//	printf("%f ",csFinal[m-1][n-1]);
					sum = 0;
				}
			}
			//printf("csFinal End\n");
			//printf("Matrix Start\n");		
			// Add vectors in parallel.
			cudaError_t cudaStatus = MatrixWithCuda(csFinal,csEnd,COUNT-1,COUNT-1);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "MatrixWithCuda failed!");
				return 1;
			}
		/*	printf("*******************************************\n");
			for (int x=0;x<COUNT-1;x++)
			{
				for (int y=0;y<COUNT-1;y++)
				{
					printf("%f ",csEnd[x*(COUNT-1)+y]);
				}
				printf("\n");
			}
			printf("*******************************************\n");
			for (int c=0;c<COUNT-1;c++)
			{
				for (int d=0;d<COUNT-1;d++)
				{
					printf("%f ",csFinal[c*(COUNT-1)+d]);
				}
				printf("\n");
			}
			printf("*******************************************\n");*/
			//cMatrix(csFinal,activeObserversSize-1,csEnd1);//求逆矩阵
		/*	for (int x=0;x<COUNT-1;x++)
			{
				for (int y=0;y<COUNT-1;y++)
				{
					printf("%f ",csEnd1[x*(COUNT-1)+y]);
				}
				printf("\n");
			}
			printf("Matrix End\n");*/
			float *usTT = (float *)malloc((activeObserversSize-1)*sizeof(float));
			for (m=0; m<activeObserversSize-1; m++)
			{
				float s =0;
				for (n = 0; n < activeObserversSize-1; n++)
				{
					//s = s + usT[n] * csEnd[n][m];
					s = s + usT[n] * csEnd[n*(activeObserversSize-1)+m];
				}
				usTT[m] = s;
			}
			float es = 0.0;
			for (m=0; m<activeObserversSize-1; m++)
			{
				es = es+usTT[m]* us[m];
			}
			//	printf("\n****假设源点为%d时估计的最大值为%f\n",graph.vertices[exti].nodeid,es);
			if (es > maxEstimator) {
				maxEstimator = es;
				maxNodeId = graph.vertices[exti].nodeid;
			}
			//printf("");
			free(us);
			free(usT);
			free(usTT);
			free(csEnd);
			free(csFinal);
		}//for  第二层，对每个节点当做源点，做出判读测出最大估计节点
		printf("\n当实际源点为%d,估计的源点为%d\n",realSourceNodeId,maxNodeId);
		if (realSourceNodeId == maxNodeId)
		{
			hit++;
		}
		maxEstimator = -100000;
		maxNodeId = 0;
	}//for第一层，对每个realNode做测试；
	printf("命中次数%d\n",hit);
	//getchar();
	printf("按任意键继续......");
	char stop;
	scanf("%s",&stop);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
   cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}
cudaError_t MatrixWithCuda(float *csFinal,float *csEnd, int height,int width)
{
	const unsigned int mem_size = sizeof(float) * width * height;
    cudaError_t cudaStatus;
	float *unitMatrix;//单位矩阵
	unitMatrix = (float*)malloc(mem_size);
	for(int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			if (i==j)
			{
				unitMatrix[i*width+j]=1.0;
			}
			else
			{
				unitMatrix[i*width+j]=0.0;
			}
		}
	}
	float *d_A;
	float *d_B;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "0cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	// cudaMalloc((void**)&d_idata, mem_size);
    cudaStatus = cudaMalloc((void**)&d_A, mem_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "1cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_B, mem_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "2cudaMalloc failed!");
        goto Error;
    }
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_A, csFinal, mem_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "3cudaMemcpy failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(d_B, unitMatrix, mem_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "3cudaMemcpy failed!");
		goto Error;
	}
    // Launch a kernel on the GPU with one thread for each element.
	//dim3 grid(1, 1);
	dim3 grid((width+BLOCK_DIM-1)/BLOCK_DIM,(width+BLOCK_DIM-1)/BLOCK_DIM);
	dim3 block(BLOCK_DIM, BLOCK_DIM);//测试时先设16
	for (int k=0;k<width;k++)
	{
		MatrixInverse_Normalized<<<grid,block>>>(d_A,d_B,width,k);
		MatrixInverse_Elimination<<<grid,block>>>(d_A,d_B,width,k);
	}
	
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "4cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(csEnd, d_B, mem_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "5cudaMemcpy failed!");
        goto Error;
    }
	//float *E = (float *)malloc(mem_size);
	//cudaStatus = cudaMemcpy(E, d_A, mem_size, cudaMemcpyDeviceToHost);
	////if (cudaStatus != cudaSuccess) {
	////	fprintf(stderr, "5cudaMemcpy failed!");
	////	goto Error;
	////}
	////for (int x=0;x<width;x++)
	////{
	////	for (int y=0;y<width;y++)
	////	{
	////		printf("%f ",E[x*width+y]);
	////	}
	////	printf("\n");
	////}
Error:
    cudaFree(d_A);
    cudaFree(d_B);
	free(unitMatrix);
    return cudaStatus;
}



