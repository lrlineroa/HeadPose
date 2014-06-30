#include "SkinDetector.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include  "opencv2/opencv.hpp"
#include "iostream"
# define PI  3.14159265358979323846
# define N_FEAT 6
# define N_IMG 90
# define N_CLASS 9
# define N_TEST 10
using namespace std;
using namespace cv;

//Mat src;

SkinDetector::SkinDetector(void)
{
//YCrCb threshold
// You can change the values and see what happens
Y_MIN  = 0;
Y_MAX  = 255;
Cr_MIN = 133;
Cr_MAX = 173;
Cb_MIN = 77;
Cb_MAX = 127;
}

SkinDetector::~SkinDetector(void)
{
}

//this function will return a skin masked image
cv::Mat SkinDetector::getSkin(cv::Mat input)
{
cv::Mat skin;
//first convert our RGB image to YCrCb
cv::cvtColor(input,skin,cv::COLOR_BGR2YCrCb);

//uncomment the following line to see the image in YCrCb Color Space
//cv::imshow("YCrCb Color Space",skin);

//filter the image in YCrCb color space
cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);

return skin;
}

void findFace(Mat & src, Mat & dst, vector<vector<Point> > &  contours, double & n_max){


    vector<Vec4i> hierarchy;
    findContours( src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
		double max=0;
		 n_max=0;
		//	cout<<"numero de contornos: "<<contours.size()<<endl;
	  for (int i =0; i<contours.size();i++){
				if(contourArea(contours[i])>max){
						max = contourArea(contours[i]);
						n_max=i;
					}					
		//			cout<<"Area "<<i<<": "<<contourArea(contours[i])<<endl;
		}
	//		cout<< "Max area: "<<max<<endl;
	    dst = Mat::zeros( src.size(), CV_8UC1 );
		  drawContours( dst, contours,n_max, 255,CV_FILLED, 8, hierarchy, 0, Point() );
			
}


void standOut_eyes(Mat & src, Mat & dst){
	  cvtColor(src, src, CV_RGB2GRAY); 
		threshold(src,dst,0,255,THRESH_BINARY| CV_THRESH_OTSU);  
//	threshold(src,dst,65,255,THRESH_BINARY);  
   	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(8,8), Point(1, 1));		
  	erode(src,dst,element);    
 	  element = getStructuringElement(CV_SHAPE_RECT, Size(5,5), Point(1, 1));	
    dilate(dst,dst,element); 

}

void searchEyes(Mat & src, Mat & dst){
	
		 
    	
	Scalar color = Scalar( 255, 0,0 );
    	vector<vector<Point> > contours;
	vector<vector<Point> > eyes2;
	double maxi=0;
	double nextMaxi=0;
	int n_maxi[2];
	int i_max;
    	vector<Vec4i> hierarchy;
    	findContours( src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//    Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
	double prevalence[contours.size()];
    	for( int i = 0; i< contours.size(); i++ )
    	{
        	
        	drawContours( dst, contours, i, Scalar(0,255,255), 1, 8, hierarchy, 0, Point());
    	}



	for( int i = 0; i< contours.size(); i++ )
    	{
        	if(contourArea(contours[i])>maxi){
			maxi=contourArea(contours[i]);
			i_max=i;
		}
        
    	}
	maxi=0;
	for( int i = 0; i< contours.size(); i++ )
        {
                if(contourArea(contours[i])>maxi && i != i_max){
                        maxi=contourArea(contours[i]);
                        n_maxi[0]=i;
			
                }  
        
        }
	maxi=0;
	 for( int i = 0; i< contours.size(); i++ )
        {
                if(contourArea(contours[i])>=maxi && i!= i_max && i != n_maxi[0]){
                        maxi=contourArea(contours[i]);
			n_maxi[1]=i;
                }  

        }
	for(int i=0;i<2;i++){
		drawContours( dst, contours, n_maxi[i], Scalar(255,0,0), 1, 8, hierarchy, 0, Point() );
	}
	
	imshow("Ojos ?",dst);		
}



void makeEllipse(vector <Point> cont, Mat & dst, float datos [][N_FEAT], int itr){

    Mat plano = Mat::zeros( dst.size(), CV_8UC3 );
  	double x=0, y=0;			
    Moments m = moments(cont,false);
		x=m.m10/m.m00;
		y=m.m01/m.m00;

	    circle( dst, Point(x, y), 3, 0, 1, CV_AA);
			circle( plano, Point(x, y), 5, Scalar(0,255,0), 1, CV_AA);
			  Mat pointsf;
        Mat(cont).convertTo(pointsf, CV_32F);
        RotatedRect box = fitEllipse(pointsf);		  

   //    	cout<<"Angulo: "<<box.angle<<endl;
				datos[itr][0]=box.angle;
				if(box.angle>=150 && box.angle <= 180 || box.angle>=0 && box.angle <=15 ){
					cout<<"ESTA MIRANDO DE FRENTE"<<endl;
			  }else if(box.angle>=20 && box.angle <= 60) {
					cout<<"ESTA MIRANDO A IZQUIERDA"<<endl;
				}else{
					cout<<"ESTA MIRANDO A DERECHA"<<endl;
			}

//				cout<<"width: "<<box.size.width<<endl;
	//			cout<<"Height: "<<box.size.height<<endl;		
		//		cout<<"Coeficiente: "<<box.size.height/box.size.width<<endl;
	
	Point2f vertices[4];
  box.points(vertices);
  for (int i = 0; i < 4; i++){
  line(dst, vertices[i], vertices[(i+1)%4], Scalar(255));
  line(plano, vertices[i], vertices[(i+1)%4], Scalar(0,0,255));
  }	

//	dst.rows = height
//	dst.cols = width
	
   line(plano, Point(x,0),Point(x,dst.rows),Scalar(255,0,0),1); //eje y
	 line(plano, Point(0,y),Point(dst.cols,y),Scalar(255,0,0),1);  // eje x

   line(dst, Point(x,0),Point(x,dst.rows),Scalar(255,0,0),1); //eje y
	 line(dst, Point(0,y),Point(dst.cols,y),Scalar(255,0,0),1);  // eje x
		//m = pendiente
	 double m1, angH;
		m1 =  (vertices[2].y - vertices[3].y)/(vertices[2].x - vertices[3].x);

		angH = atan(m1)*180/PI ;
		//cout<<"Pendiente recta M1: "<<m1<<endl;
    cout<<"Angulo respecto a la horizontal: "<<angH<<endl;	
		datos[itr][1]=angH;	
		int rows =dst.rows;
		int cols = dst.cols;
		float px_c1=0.0,px_c2=0.0,px_c3=0.0,px_c4=0.0;

		for(int i=0;i<rows;i++){
			uchar *p = dst.ptr(i);
			for(int j=0;j<cols;j++){
				if(i < x && j > y && p[j] == (uchar)255){ // cuadrante I 
					px_c1++;											
				}else if(i < x && j < y && p[j] == (uchar)255){// cuadrante II
					px_c2++;
				}else if(i > x && j < y && p[j] == (uchar)255){// cuadrante III
					px_c3++;
				}else if(i > x && j > y && p[j] == (uchar)255){// cuadrante IV			
					px_c4++;			
		}
	}
}
				datos[itr][2]=px_c1;
				datos[itr][3]=px_c2;
				datos[itr][4]=px_c3;
				datos[itr][5]=px_c4;

			//	cout<<"Total Px Cuadrante I: "<<px_c1<<endl;
			//	cout<<"Total Px Cuadrante II: "<<px_c2<<endl;
			//	cout<<"Total Px Cuadrante III: "<<px_c3<<endl;
		//		cout<<"Total Px Cuadrante IV: "<<px_c4<<endl;
					imshow("Cara",dst);
					imshow("Plano",plano);
}


void make_dates(string src, float datos[][N_FEAT], int num) {
 Mat dst,mask;
 string dir;
//		Moments m; 
//		double hu [8];
// if(num == N_IMG){


//}
 for (int i = 0; i < num; i++) {
  	 Mat img;
		 if(i>=0 && i<=9){
			dir= src;
			dir +='a';
			dir += char(i+'0');
			dir += ".jpg";
     img = imread(dir,1);
		 }else if(i>9 && i<=19){
		 dir =src;
		 dir +='b';
     dir += char(i-10+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 
		 }else if(i>19 && i<=29){
		 dir =src;
		 dir +='c';
     dir += char(i-20+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 
		 }else if(i>29 && i<=39){
		 dir =src;
		 dir +='d';
     dir += char(i-30+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 				 
		 }else if(i>39 && i<=49){
		 dir =src;
		 dir +='e';
     dir += char(i-40+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 				 
		 }else if(i>49 && i<=59){
		 dir =src;
		 dir +='f';
     dir += char(i-50+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 				 
		 }else if(i>59 && i<=69){
		 dir =src;
		 dir +='g';
     dir += char(i-60+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 				 
		 }else if(i>69 && i<=79){
		 dir =src;
		 dir +='h';
     dir += char(i-70+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 				 
		 }else if(i>79 && i<=89){
		 dir =src;
		 dir +='i';
     dir += char(i-80+'0');
     dir += ".jpg";
     img = imread(dir,1);
  //   imshow(dir,img); 				 
		 }
// }

//  double x,y;	
	double x,y,max;	
	vector<vector<Point> > contours;
//	Mat src;
//	src = imread("images/img_1.jpg");
	Mat dst=img.clone();
  Mat eyes= Mat::zeros( img.size(), CV_8UC3 );
// src se llama img
	imshow("Original Image",img);
	SkinDetector mySkinDetector;
	Mat skinMat;
	skinMat= mySkinDetector.getSkin(img);

	findFace(skinMat,dst,contours,max);			
	Mat mask(img.rows, img.cols, CV_8UC3);
	mask.setTo(Scalar(0,0,0));
	img.copyTo(mask, dst);
	standOut_eyes(mask, mask);
  searchEyes(mask, eyes);
	findFace(mask,dst,contours,max);
	img.copyTo(mask, dst);
	standOut_eyes(mask, mask);
 	makeEllipse(contours[max],mask,datos,i);
	double hu [1];
//	Moments m;
//	m=moments(cara,false);	


	
//	datos[i][1]=m.m00;






//	makeEllipse();
   //para separar la cara
	
 }//end for
}



int histograma(Mat & src)
{
	int tresh =0;
  int histSize = 256;
	float range[] = { 0, 256 } ;
  const float* histRange = { range };
	float diff =0, max_diff=0;  
  bool uniform = true; bool accumulate = false;
  Mat hist;
	calcHist( &src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
  normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

 //Draw
  for( int i = 1; i < histSize; i++ )
  {
    line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                      Scalar( 255, 255, 0), 2, 8, 0  );
 }

	//imshow("calcHist Demo", histImage );
	
}


int main()
{
	double x,y;	
	Mat src;
	const int K = 10;
  	float datos[N_IMG][N_FEAT];
	float tests[N_TEST][N_FEAT];
	int clasif[N_IMG]={1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,
										 4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,
										 7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9};	

  	float testClasf[N_TEST]={1,2,1,3,3,7,5,7,8,9};
 //                        4,2,2,2,4,5,2,8,9,0

	make_dates("images/img_",datos,N_IMG);
	make_dates("tests/test_",tests,N_TEST);

	Mat data=Mat(N_IMG,N_FEAT,CV_32FC1,datos);
	Mat respuesta= Mat(N_IMG,1,CV_32SC1,clasif);

	Mat testingData = Mat(N_TEST,N_FEAT,CV_32FC1,tests);
	Mat testClasific = Mat(N_TEST,1,CV_32FC1,testClasf);


 	CvKNearest* knn = new CvKNearest;
	knn->train(data, respuesta,Mat(), false, K);

	Mat test_sample;
        int k;
    	int correct_class = 0;
    	int wrong_class = 0;
    	int false_positives [N_CLASS] = {0,0,0,0,0,0,0,0,0};
	int clases[N_CLASS]={1,2,3,4,5,6,7,8,9};
	float resultNode;

	Mat nearest = Mat( 1, K, CV_32FC1);

	for (int tsample = 0; tsample < N_TEST; tsample++){
			test_sample = testingData.row(tsample);
			resultNode = knn->find_nearest(test_sample,K,0,0, & 				nearest, 0);
			cout<<"La muestra "<<tsample<<" esta en la clase: "<<clases[(int)(resultNode)]<<endl;				
            		if (fabs(resultNode - testClasific.at<float>(tsample, 0)) >= FLT_EPSILON){
                		wrong_class++;
                		false_positives[(int) resultNode]++;

           		}else{
                		correct_class++;
            		}
  	}
	cout<<"Clases Correctas: "<<(double) (correct_class*100)/N_TEST<<endl;
	cout<<"Clases Equivocadas: "<< (double) (wrong_class*100)/N_TEST<<endl;

        for (int i = 0; i < N_CLASS; i++){
		cout<<"Falsos positivos clase "<<clases[i]<<": "<<false_positives[i]<<" Porcentaje: "<<(double)(false_positives[i]*100)/N_TEST<<endl;
        }

//	CvNormalBayesClassifier *bayes = new CvNormalBayesClassifier;	
//		bayes->train(data, respuesta,Mat(),Mat(),false);



/*  Mat test_sample;
    int k;
    int correct_class = 0;
    int wrong_class = 0;
    int false_positives [N_CLASS] = {0,0,0,0,0,0,0,0,0};
	  int clases[N_CLASS]={1,2,3,4,5,6,7,8,9};

                         
		float result=0;

    for (int tsample = 0; tsample < N_TEST; tsample++){

            test_sample = testingData.row(tsample);
            result = bayes->predict(test_sample);
						cout<<"La muestra "<<tsample<<" esta en la clase: "<<clases[(int)(result)]<<endl;
						cout<<"REsult "<<(result)<<endl;
            if (fabs(result - testClasific.at<float>(tsample, 0))>= FLT_EPSILON){
                wrong_class++;
                false_positives[((int) result)]++;

						}else{
							 correct_class++;
						}
				}
					cout<<"Clases Correctas: "<<(double) (correct_class*100)/N_TEST<<endl;
					cout<<"Clases Equivocadas: "<< (double) (wrong_class*100)/N_TEST<<endl;

	        for (int i = 0; i < N_CLASS; i++){					
           	cout<<"Falsos positivos clase "<<clases[i]<<": "<<false_positives[i]<<" Porcentaje: "<<(double)(false_positives[i]*100)/N_TEST<<endl;
	        }


*/


 for(int i=0; i< N_IMG; i++){
		cout<<"Imagen "<<(i+1)<<": ";
		for(int j =0; j<N_FEAT;j++){
			cout<<datos[i][j]<<", ";
		}
		cout<<"]"<<endl;
	}

/*	Mat trainData = Mat(N_IMG,N_FEAT,CV_32FC1,datos); 
	Mat clusters = Mat(N_IMG,1, CV_32FC2);
	Mat centers(9, N_FEAT, CV_32FC2);

	kmeans(trainData,9,clusters, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ), 10, KMEANS_PP_CENTERS,centers);
	

		int wrong_cluster=0;
		int correct_cluster=0;
		for (int tsample = 0; tsample < N_IMG; tsample++){
				
			if(clusters.at<float>(tsample,1) == clasif[tsample]){
				correct_cluster++;
			}else{
				wrong_cluster++;
			}
	  }

		cout<<"Labels: \n"<<clusters<<endl;
		cout<<"Puntos: \n"<<centers<<endl;
	

					cout<<"Clusters Correctos: "<<(double) (correct_cluster*100)/N_IMG<<endl;
					cout<<"Clusters Equivocados: "<< (double) (wrong_cluster*100)/N_IMG<<endl;




*/
	


waitKey(0);
//}
return 0;
}
