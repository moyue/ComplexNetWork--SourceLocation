
//#define MAX_NAME 10
#define MAX_VERTEX_NUM 200      //顶点数
#define MAX_EDGE_NUM 300		//边数

//#define DIFF_EDGE_NUM 20000          //diffuion使用的临时数组大小
typedef int InfoType;				// 存放网的权值 
//typedef char VertexType[MAX_NAME];	// 字符串类型 
//typedef enum{DG,AG}GraphKind; // {有向图,无向图} 



typedef struct Arc
{
	int arcId;
	int arcsourceId;
	int arctargerId;
	InfoType Weight;
	InfoType tmpWeight;
}ArcList[MAX_EDGE_NUM];

typedef struct ArcNode
{
	int arcId;
	int arcsourceId; //一趟的arcsourceID应该等于VNode的nodeid
	int arctargerId;
	int adjvex;					// 该弧所指向的顶点的位置 
	struct ArcNode *nextarc;	// 指向下一条弧的指针 			 
	//InfoType Weight;			// 图的权值
	//InfoType tmpWeight;
}ArcNode;	// 表结点 

typedef struct VNode
{
	//VertexType label;			// 顶点信息
	int degree;					//节点的度
	int nodeid;
	int time;			//记录传播时间
	bool isActive;
	int direction;	 
	int parent;
	int plength;
	ArcNode *firstarc;			// 第一个表结点的地址,指向第一条依附该顶点的弧的指针 
}VNode,AdjList[MAX_VERTEX_NUM];// 头结点 

typedef struct
{	
	ArcList arctices;
	AdjList vertices;
	int vexnum,arcnum;	// 图的当前顶点数和弧数 
	//int kind;			// 图的种类标志 
}ALGraph;

//ALGraph graph;

int  GetNodeIndex(ALGraph G,int nodeid)//根据节点ID返回在数组中的索引值
{
	int i;
	for(i=0;i<G.vexnum;++i)
		if(G.vertices[i].nodeid==nodeid)
			return i;
	return -1;

}
/************************************************************************/
/* 将节点相邻的边的ID加入数组     */  
/************************************************************************/
int *getNodeEdges(ALGraph *graph,int nodeId)
{
	int nodeIndex = GetNodeIndex(*graph,nodeId);
	int nodeDegree = graph->vertices[nodeIndex].degree;
	int *edges;
	// 	if (edges != NULL)
	// 	{
	// 		free(edges);
	// 	}
	edges = (int *)malloc(nodeDegree * sizeof(int));
	ArcNode *p;
	p = graph->vertices[nodeIndex].firstarc;
	int i = 0;
	while(p)
	{
		edges[i] = p->arcId;
		p=p->nextarc;
		i++;
	}
	free(p);
	return edges;

}
int GetEdgeId(ALGraph graph,int sourceid,int targetid)
{
	int nodeIndex = GetNodeIndex(graph,sourceid);
//	int nodeDegree = graph.vertices[nodeIndex].degree;
	ArcNode *p;
	p = graph.vertices[nodeIndex].firstarc;
	int arcID;
	while(p)
	{
		if (p->arctargerId == targetid)
		{
			arcID = p->arcId;
			break;
			//return arcID;
		}
		p=p->nextarc;
	}
	return arcID;
}
int CreateGraph(ALGraph *graph)//建图最好结构数组编号为nodeid//查找快速；但可能不是，独立点问题
{// 只按无向图来设计

	FILE *fpNode;
	FILE *fpEdge;
	//char charEnd;

	if ((fpEdge=fopen("C:\\BA_5_Edges.csv","r"))==NULL)
	{
		printf("no file\n");
		exit(0);
	}
	if ((fpNode=fopen("C:\\BA_5_Nodes.csv","r"))==NULL)
	{
		printf("no file\n");
		exit(0);
	}
	int nodeNum = 0;
	while(!feof(fpNode))
	{			
		int tmpNodeId;
		fscanf(fpNode,"%d",&tmpNodeId);
		printf("%d\n",tmpNodeId);
		// **不用了，在此以得到的节点ID为图中数组的ID利于随机存储（问题《若节点不连续可能有中间节点未赋值）；不行还是按顺序存吧

		graph->vertices[nodeNum].nodeid = tmpNodeId;
		graph->vertices[nodeNum].firstarc = 0;
		graph->vertices[nodeNum].degree = 0;
		graph->vertices[nodeNum].direction = -1;
		graph->vertices[nodeNum].isActive = false;
		graph->vertices[nodeNum].time = 1;
		graph->vertices[nodeNum].parent = -1;
		nodeNum++;
	}
	graph->vexnum = nodeNum-1;
	int edgeNum = 0;
	while(!feof(fpEdge))
	{	
		int sourceid;
		int targetid;
		fscanf(fpEdge,"%d %d",&sourceid,&targetid);
		printf("%d,%d\n",sourceid,targetid);
		graph->arctices[edgeNum].arcId = edgeNum;
		graph->arctices[edgeNum].arcsourceId = sourceid;
		graph->arctices[edgeNum].arctargerId = targetid;
		graph->arctices[edgeNum].Weight = 1;
		graph->arctices[edgeNum].tmpWeight =1;
		edgeNum++;
	}
	graph->arcnum = edgeNum-1;
	fclose(fpNode);
	fclose(fpEdge);

	int i,j,k;
	int nodeindex;
	//	int w;		// 权值 
	//VertexType va,vb;
	ArcNode *p;
	int degree;
	for(k = 0;k < graph->arcnum; ++k)	// 构造表结点链表 
	{
		i =	graph->arctices[k].arcsourceId;// 弧头 
		j =	graph->arctices[k].arctargerId;// 弧尾
		p = (ArcNode*)malloc(sizeof(ArcNode));
		p->arcId= graph->arctices[k].arcId;
		p->arcsourceId = i;
		p->arctargerId = j;
		//p->Weight =(*graph).arctices[k].Weight;
		p->adjvex =j;
		nodeindex = GetNodeIndex(*graph,i); 
		p->nextarc = graph->vertices[nodeindex].firstarc; // 插在表头 
		graph->vertices[nodeindex].firstarc = p;

		degree = graph->vertices[nodeindex].degree;//设置节点的度
		graph->vertices[nodeindex].degree = degree + 1;

		//无向图，还得在j上加一条边
		p = (ArcNode*)malloc(sizeof(ArcNode));
		p->arcId=graph->arctices[k].arcId;//要与上一个无向的使用相同的边ID
		p->arcsourceId = j;
		p->arctargerId = i;
		p->adjvex = i;
		nodeindex = GetNodeIndex(*graph,j);
		p->nextarc = graph->vertices[nodeindex].firstarc; // 插在表头 
		graph->vertices[nodeindex].firstarc = p;
		degree = graph->vertices[nodeindex].degree;//设置节点的度
		graph->vertices[nodeindex].degree = degree + 1;
	}
	return 1;
}
//	输出图的邻接表G。
void Display(ALGraph G)
{
	int i;
	ArcNode *p;
	printf("%d个顶点：\n",G.vexnum);
	for(i = 0; i < G.vexnum; ++i)
		printf("%d ",G.vertices[i].nodeid);
	printf("\n%d条弧(边):\n", G.arcnum);
	for(i = 0; i < G.vexnum; i++)
	{
		p = G.vertices[i].firstarc;
		while(p)
		{
			// 			if (p->arcsourceId<p->arctargerId)
			// 			{
			// 				printf("%d-%d ",p->arcsourceId,p->arctargerId);
			// 			}
			// 			p=p->nextarc;


			printf("%d-%d-%d ",p->arcId,p->arcsourceId,p->arctargerId);

			p=p->nextarc;
		}
		printf("\n");
		printf("度为%d\n", G.vertices[i].degree);
	}
}
