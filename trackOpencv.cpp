#include<iostream>
#include<string>
#include <algorithm>
#include<opencv2/opencv.hpp>

#include"Entity.h"
using namespace std;
using namespace cv;
ostream& operator<<(ostream& out, Entity& en)    //进来后又出去

{
	cout << "id :" << en.id << endl;
	cout << "stat :" << en.stat << endl;
	cout << "sunRect :";
	for (auto rect : en.sunRect)
	{
		cout << rect << endl;
	}
	return out;

}

string getTime(double start, const string name, const string ext = "ms")
{
	double end = (double)clock();
	int use = (end - start);
	string s;
	if (ext == "ms") s = name + " time is : " + to_string(use) + "ms\n";
	if(ext == "s") s = name + " time is : " + to_string(use / 1000) + "s\n";
	if (ext == "min")
	{
		use = (end - start) / 1000.0;
		int minute = use / 60;
		int second = use % 60;
		s = name + " time is :" + to_string(minute) + "min" + to_string(second) + "s\n";
	}
	return s;
}

class Road
{
public:
	static int num;
	friend class Video;
	Road(vector<Point> points, int trajectoryDis = 5, int trajectoryNum = 5, int edge = 100, bool show = false, float iou_low = 0.5, \
		int minLength = 300, int history = 50, int contourArea = 4000, float ratio_low = 1, \
		float ratio_high = 5, int LaneNumber = 3 ) :
		points(points), trajectoryDis(trajectoryDis), trajectoryNum(trajectoryNum), edge(edge), show(show),\
		minLength(minLength), iou_low(iou_low), history(history), contourArea(contourArea), \
		ratio_low(ratio_low), ratio_high(ratio_high), LaneNumber(LaneNumber)
	{ 
		pMOG2 = createBackgroundSubtractorMOG2(200);
		roadId = ++num;
		if(show) namedWindow(to_string(roadId));
	}

	void getRotmat()
	{
		//获取仿射变换矩阵
		rect = minAreaRect(points);
		double x0, y0, alpha;
		x0 = rect.center.x;
		y0 = rect.center.y;
		alpha = -rect.angle * CV_PI / 180;
		Mat rot_mat_tmp(2, 3, CV_32FC1);
		rot_mat_tmp = getRotationMatrix2D(rect.center, rect.angle, 1.0);
		rot_mat = rot_mat_tmp;
		Mat invert_mat_tmp(2, 3, CV_32FC1);
		invertAffineTransform(rot_mat, invert_mat_tmp);
		invert_mat = invert_mat_tmp;
		//获取缩放尺度
		if (rect.size.width > rect.size.height)
		{
			scale = rect.size.height * 1.0 / minLength;
		}
		else
		{
			scale = rect.size.width * 1.0 / minLength;
		}
	}
	
	void proc(Mat &m)
	{
		if(++mNum == 1)
		{
			getRotmat();
		}
		
		getRoad(m);
		detect();
		visual();
		getBack(m);
	}
private:
	vector<vector<Point>> getContours()
	{
		Mat binary;
		threshold(fgMaskMOG2, binary, 244, 255, THRESH_BINARY);
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		Mat erode_t;
		erode(binary, erode_t, element);
		Mat dilate_t;
		element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		dilate(erode_t, dilate_t, element);
		
		int i = 0;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		
		for (i = 0; i < LaneNumber; i++)
		{
			Mat subdilate_t = Mat::zeros(dilate_t.rows, dilate_t.cols, dilate_t.depth());
			if (dilate_t.cols > dilate_t.rows)
			{
				int x1 = 0;
				int y1 = i * dilate_t.rows / LaneNumber;
				int w = dilate_t.cols;
				int h = dilate_t.rows / LaneNumber;
				Rect rect = Rect(x1, y1, w, h);
				dilate_t(rect).copyTo(subdilate_t(rect));
				
				vector<vector<Point>> subcontours;
				vector<Vec4i> hierarchy; 
				findContours(subdilate_t, subcontours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
				contours.insert(contours.end(), subcontours.begin(), subcontours.end());
			}
		}
		
		return contours;
	}
	void getRoad(Mat m)
	{
		//获取道路和背景分离的图片
		Mat maskRoad = Mat(m.rows, m.cols, CV_8UC3, Scalar(0, 0, 0));
		vector<vector<Point> > vpts;
		vpts.push_back(points);
		fillPoly(maskRoad, vpts, Scalar(255, 255, 255), 8, 0);
		Mat mRoad;
		bitwise_and(m, maskRoad, mRoad);
		Mat maskBack;
		bitwise_not(maskRoad, maskBack);
		bitwise_and(m, maskBack, mBack);
		//道路校正
		Size dst_sz(m.rows, m.cols);
		warpAffine(mRoad, roiInm, rot_mat, dst_sz);
		//提取道路
		int x1 = int(rect.center.x - rect.size.width / 2);
		int y1 = int(rect.center.y - rect.size.height / 2);
		Mat roadOri = roiInm(Rect(x1, y1, int(rect.size.width), int(rect.size.height)));
		
		int w = int(roadOri.cols / scale);
		int h = int(roadOri.rows / scale);
		resize(roadOri, road, Size(w, h));
			
	}

	void getBack(Mat& m)
	{
		//road_FIRST = cv2.resize(road_FIRST, ((int)(x2_FIRST - x1_FIRST), (int)(y2_FIRST - y1_FIRST)), interpolation = cv2.INTER_CUBIC)

		//masked1_FIRST[y1_FIRST:y2_FIRST, x1_FIRST : x2_FIRST] = road_FIRST
		//masked1_FIRST = cv2.warpAffine(masked1_FIRST, invert_rot_mat_FIRST, (cols, rows))
		int w = int(road.cols * scale);
		int h = int(road.rows * scale);
		Mat roadOri;
		resize(road, roadOri, Size(w, h));
		Rect rect_t = Rect(rect.center.x - w/2, rect.center.y - h/2, w, h);
		roadOri.copyTo(roiInm(rect_t));
		warpAffine(roiInm, roiInm, invert_mat, Size(roiInm.rows, roiInm.cols));
		bitwise_or(roiInm, mBack, m);
	}


	void getEnList(vector<vector<Point>> contours)
	{
		vector<int> updateId;
		vector<Entity> newenList;
		for (auto c : contours)
		{
			Rect r = boundingRect(c);
			bool flag = false;
			if (r.width * r.height > contourArea)
			{
				if (dir == "left" || dir == "right")
				{
					float ratio = r.width * 1.0 / r.height;
					if (ratio > ratio_low && ratio < ratio_high)
					{
						flag = true;
					}
				}
				else if (dir == "top" || dir == "down")
				{
					float ratio = r.height * 1.0 / r.width;
					if (ratio > ratio_low && ratio < ratio_high)
					{
						flag = true;
					}
				}
				else
				{
					flag = true;
				}
			}
			if (flag)
			{
				if (enList.size() == 0)
				{
					newenList.push_back(Entity(enId, r));
					updateId.push_back(enId);
					enId++;
					
				}
				else
				{
					int max_index = -1;
					float max_iou = -1.0;
					for (int i = 0; i < enList.size(); i++)
					{
						float iou = enList[i].overlap(r);
						if (iou > iou_low&& iou > max_iou)
						{
							max_iou = iou;
							max_index = i;
						}
					}
		
					if (max_index == -1)
					{
						newenList.push_back(Entity(enId, r));
						updateId.push_back(enId);
						enId++;
					}
					else
					{
						enList[max_index].add(r);
						updateId.push_back(enList[max_index].id);
					}

				}
			}	
		}

		enList.insert(enList.end(), newenList.begin(), newenList.end());
		for (vector<Entity>::iterator en = enList.begin(); en != enList.end();)
		{
			if (find(updateId.begin(), updateId.end(), en->id) == updateId.end())
			{
				en = enList.erase(en);
			}
			else
			{
				en++;
			}
		}
	}
	
	void detect()
	{
		pMOG2->apply(road, fgMaskMOG2);
		vector<vector<Point>> contours = getContours();
		if (mNum < history)
		{
			getEnList(contours);
		}
		else if(mNum == history)
		{
			int left, right, top, down;
			left = right = top = down = 0;
			for (auto en : enList)
			{
				if (en.sunRect.size() > trajectoryNum)
				{
					Rect last = *(en.sunRect.rbegin());
					Rect first = *(en.sunRect.begin());
					int dx = last.x - first.x;
					int dy = last.y - first.y;
					
					if (road.cols > road.rows)
					{
						if (dx > 0) right++;
						if (dx < 0) left++;
					}
					else
					{
						if (dy > 0) down++;
						if (dy < 0) top++;
					}
				}
			}
			if (road.cols > road.rows)
			{
				if (left > right) dir = "left";
				else dir = "right";
			}
			else
			{
				if (down > top) dir = "down";
				else dir = "top";
			}
		}
		else
		{
			getEnList(contours);
			
			for(auto en = enList.begin(); en != enList.end();)
			{
				Rect last = *(en->sunRect.rbegin());
				Rect first = *(en->sunRect.begin());
				if (dir == "right")
				{
					if (last.x + edge >= road.cols && first.x + edge < road.cols && en->sunRect.size() > trajectoryNum && last.x - first.x > trajectoryDis)
					{
						pass++;
						en = enList.erase(en);
						cout << "right passNum :" << pass << endl;
					}
					else
					{
						
						en++;
					}
				}
				if (dir == "left")
				{
					if (last.x - edge <= 0 && first.x - edge > 0 && en->sunRect.size() > trajectoryNum && first.x - last.x > trajectoryDis)
					{
						pass++;
						en = enList.erase(en);
						cout << "left passNum :" << pass << endl;
					}
					else
					{
						en++;
					}
				}
				if (dir == "top")
				{
					if (last.y - edge <= 0 && first.y - edge > 0 && en->sunRect.size() > trajectoryNum && first.y - last.y > trajectoryDis)
					{
						pass++;
						en = enList.erase(en);
						cout << "passNum :" << pass << endl;
					}
					else
					{
						en++;
					}
				}
				if (dir == "down")
				{
					if (last.y + edge >= road.rows && first.y - edge < road.rows && en->sunRect.size() > trajectoryNum&& last.y - first.y > trajectoryDis)
					{
						pass++;
						en = enList.erase(en);
						cout << "passNum :" << pass << endl;
					}
					else
					{
						en++;
					}
				}
			}
		}
	}

	

	void visual()
	{
		for (auto en : enList)
		{
			rectangle(road, *(en.sunRect.rbegin()), Scalar(0, 255, 0), 2, LINE_8, 0);
			putText(road, to_string(en.id), Point((*(en.sunRect.rbegin())).x, (*(en.sunRect.rbegin())).y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1, 8, false);
		}
		Point p1, p2;
		if (dir == "right")
		{
			p1 = Point(road.cols - edge, 0), p2 = Point(road.cols - edge, road.rows);
		}
		if (dir == "left")
		{
			p1 = Point(edge, 0), p2 = Point(edge, road.rows);
		}
		if (dir == "top")
		{
			p1 = Point(0, edge), p2 = Point(road.cols, edge);
		}
		if (dir == "dwon")
		{
			p1 = Point(0, road.rows - edge), p2 = Point(road.cols, road.rows - edge);
		}
		line(road, p1, p2, cv::Scalar(0, 255, 0), 3, 4);
		if (show)
		{
			imshow(to_string(roadId), road);
			waitKey(1);
		}
	}
	
private:
	int minLength;
	float iou_low;
	int history;
	int contourArea;
	float ratio_low;
	float ratio_high;
	int LaneNumber;
	int trajectoryDis;
	int trajectoryNum;
	int edge;
	//道路参数
	int mNum = 0;
	float scale;
	Mat road, mBack, roiInm;
	string dir;
	vector<Point> points;
	RotatedRect rect;
	Mat rot_mat;
	Mat invert_mat;
	Ptr<BackgroundSubtractor> pMOG2;
	Mat fgMaskMOG2;
	//实体参数
	int enId = 0;
	vector<Entity> enList;
	int pass;

	int roadId;
	bool show;
};
int Road::num = 0;

class Video
{
public:
	Video(string sourceFile, string procFile = "", bool show = true, bool save = true) : sourceFile(sourceFile), procFile(procFile), show(show), save(save) {}

	void open()
	{
		cap = VideoCapture(sourceFile);
		if (!cap.isOpened())
		{
			cout << "open video error!";
			return;
		}

		if (procFile != "")
		{
			int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
			int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
			writer = VideoWriter(procFile, CV_FOURCC('M', 'J', 'P','G'), cap.get(CAP_PROP_FPS), Size(frame_width, frame_height));
		}
	}
	bool read(Mat& m)
	{
		return cap.read(m);
	}
	void write(Mat& m)
	{
		writer.write(m);
	}
	int visual(Mat& m, initializer_list<Road> il)
	{
		int w, h;
		w = 220, h = 50;
		int i = 0;
		for (auto ptr = il.begin(); ptr != il.end(); ptr++)  //类似于容器的操作
		{
			Mat zero = Mat::zeros(h, w, CV_8UC3);
			string text = " road :" + to_string(ptr->roadId);
			text +=" passed: " + to_string(ptr->pass);
			putText(zero, text, Point(5, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1, 8, false);
			Rect rect;
			if(i == 0)  rect = Rect(0, 0, w, h);
			else  rect = Rect(0, 20 + h * i, w, h);
			i++;
			zero.copyTo(m(rect));
		}
		if (show)
		{
			imshow("processed", m);
			if(waitKey(1)>0) return 0;
		}
		if (save) write(m);
		return 1;
	}
	void release()
	{
		cap.release();
		if (procFile != "")
		{
			writer.release();
		}
	}

private:
	string sourceFile;
	string procFile;
	bool show;
	bool save;
	VideoCapture cap;
	VideoWriter writer;
};


int main()
{
	double time_Start = (double)clock();  //程序运行计时开始
	
	string sourceFile = "..//..//video//test2.avi";
	string procFile = "..//..//video//proc_test2.avi";
	Video v = Video(sourceFile, procFile,true);
	int points_t1[][2] = { { 345, 302 }, { 327, 259 }, { 465, 205 }, { 482, 248 } };
	int points_t2[][2] = { { 196, 426 } ,{ 174, 373 }, { 358, 298 },{ 379, 350 } };
	vector<Point> points;
	for (int i = 0; i < sizeof(points_t1) / sizeof(points_t1[0]); i++)
	{
		points.push_back(Point(points_t1[i][0], points_t1[i][1]));
	}
	Road r1 = Road(points);
	points.clear();
	for (int i = 0; i < sizeof(points_t2) / sizeof(points_t2[0]); i++)
	{
		points.push_back(Point(points_t2[i][0], points_t2[i][1]));
	}
	Road r2 = Road(points, 3, 3, 100, false);
	v.open();
	Mat m;
	double start_total = 0, start_read = 0, start_proc_r1 = 0, start_proc_r2 = 0, start_vis = 0;
	while (v.read(m))
	{
		//cout << getTime(start_read, "read", "ms") ;
		//start_proc_r1 = (double)clock();
		r1.proc(m);
		//cout << getTime(start_proc_r1, "proc_r1", "ms");
		//start_proc_r2 = (double)clock();
		r2.proc(m);
		//cout << getTime(start_proc_r2, "proc_r2", "ms");
		//start_vis = (double)clock();
		int res = v.visual(m, { r1, r2 });
		//cout << getTime(start_vis, "visual ", "ms");
		if (res == 0) break;
		//start_read = (double)clock();
	}
	cout << getTime(start_total, "total", "min");
	v.release();
	return 1;
}
