#include <ctime>
#include <cmath>
#include <string>
#include <iostream>
#include "ncnn/net.h"
#include <opencv2/opencv.hpp>

/*
* brief 模型配置结构体
* iInputImgH	输入图像高度
* iInputImgW	输入图像宽度
* afMean	图像处理均值
* afNorm	图像处理方差
* flColSampleW	没看懂原作者的注释QAQ
* aiRowAnchor	先验Y坐标
* pacModelbinPath	模型文件路径
* pacModelParamPath	模型文件路径
*/
typedef struct
{
    int iInputImgH;
    int iInputImgW;
    float afMean[3];
    float afNorm[3];
    float flColSampleW;
    int aiRowAnchor[18];
    std::string pacModelbinPath;
    std::string pacModelParamPath;
}MODEL_CONFIG;

/*
* brief 最终X坐标结果结构体，包含4条车道线，每条车道线至多18个点
*/
typedef struct
{
    float afLaneRslt[18][4];
}LANE_RESULT;

/*
* brief 车道线检测类
*/
class clModel
{
private:
    int iInputImgH;
    int iInputImgW;
    float afMean[3];
    float afNorm[3];
    float flColSampleW;
    int aiRowAnchor[18];
    clock_t startTime, endTime;

    ncnn::Net ncnnModel;
    ncnn::Mat ncnnMatIN;
    ncnn::Mat ncnnMatOUT;

public:
    ~clModel(void);
    LANE_RESULT stLaneRslt;
    void GetModelRslt(cv::Mat cvMatImg);
    clModel(MODEL_CONFIG* pstModelConfig);
};

/* 初始化类 */
clModel::clModel(MODEL_CONFIG* pstModelConfig)
{
    memcpy(afMean, pstModelConfig->afMean, sizeof(afMean));
    memcpy(afNorm, pstModelConfig->afNorm, sizeof(afNorm));
    iInputImgH = pstModelConfig->iInputImgH;
    iInputImgW = pstModelConfig->iInputImgW;
    memcpy(aiRowAnchor, pstModelConfig->aiRowAnchor, sizeof(aiRowAnchor));
    flColSampleW = pstModelConfig->flColSampleW;
    ncnnModel.load_param(pstModelConfig->pacModelParamPath.c_str());
    ncnnModel.load_model(pstModelConfig->pacModelbinPath.c_str());
}

clModel::~clModel()
{
    ncnnModel.clear();
    ncnnMatIN.release();
    ncnnMatOUT.release();
}

/* 模型后处理关键函数1 */
void clModel::GetModelRslt(cv::Mat cvMatImg)
{
    int iTemp;
    ncnn::Extractor ex = ncnnModel.create_extractor();
    ncnnMatIN = ncnn::Mat::from_pixels_resize(cvMatImg.data, ncnn::Mat::PIXEL_BGR2RGB, cvMatImg.cols, cvMatImg.rows, iInputImgW, iInputImgH);
    ncnnMatIN.substract_mean_normalize(afMean, afNorm);

    startTime = clock();
    ex.input("input", ncnnMatIN);
    ex.extract("output", ncnnMatOUT);
    
    int c = ncnnMatOUT.c;
    int w = ncnnMatOUT.w;
    int h = ncnnMatOUT.h;

    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            iTemp = 0;
            for(int k = 1; k < c; k++)
            {
                if(ncnnMatOUT[k*h*w + i*w + j] > ncnnMatOUT[iTemp*h*w + i*w + j])
                {
                    iTemp = k;
                }
                else
                {
                    continue;
                }
            }
            if(iTemp == (c-1))
            {
                stLaneRslt.afLaneRslt[i][j] = 0;
            }
            else
            {

                stLaneRslt.afLaneRslt[i][j] = ((float)iTemp + 1) * flColSampleW;
            }
        }
    }



    endTime = clock();
    std::cout<<"time is: "<<(double)(endTime - startTime) / CLOCKS_PER_SEC<<"s"<<std::endl;

    // for(int i = 0; i < h; i++)
    // {
    //     for(int j = 0; j < w; j++)
    //     {
    //         std::cout<<stLaneRslt.afLaneRslt[i][j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    ex.clear();
    ncnnMatOUT.release();
}

/* 模型后处理关键函数2 */
/*
void clModel::GetModelRslt(cv::Mat cvMatImg)
{
    int iTemp;
    ncnn::Extractor ex = ncnnModel.create_extractor();
    ncnnMatIN = ncnn::Mat::from_pixels_resize(cvMatImg.data, ncnn::Mat::PIXEL_BGR2RGB, cvMatImg.cols, cvMatImg.rows, iInputImgW, iInputImgH);
    ncnnMatIN.substract_mean_normalize(afMean, afNorm);

    startTime = clock();
    ex.input("input", ncnnMatIN);
    ex.extract("output", ncnnMatOUT);
    
    int c = ncnnMatOUT.c;
    int w = ncnnMatOUT.w;
    int h = ncnnMatOUT.h;

    float *prob = new float[h*w*c];
    float *prob_sum = new float[h*w];
    memset(prob, 0, h*w*c*sizeof(float));
    memset(prob_sum, 0, h*w*sizeof(float));

    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            for(int k = 0; k < c-1; k++)
            {
                prob_sum[i*w+j] += std::exp(ncnnMatOUT[k*h*w + i*w + j]);
            }
        }
    }

    for(int i = 0; i < c-1; i++)
    {
        for(int j = 0; j < h; j++)
        {
            for(int k = 0; k < w; k++)
            {
                prob[i*h*w + j*w + k] = std::exp(ncnnMatOUT[i*h*w + j*w + k]) / prob_sum[j * w + k];
            }
        }
    }

    for(int i = 0; i < c-1; i++)
    {
        for(int j = 0; j < h; j++)
        {
            for(int k = 0; k < w; k++)
            {
                prob[i*h*w + j*w + k] = prob[i*h*w + j*w + k] * (float)(i+1);
            }
        }
    }

    float *loc = new float[h*w];
    memset(loc, 0, h*w*sizeof(float));
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            for(int k = 0; k < c - 1; k++)
            {
                loc[i*w + j] += prob[k*h*w + i*w + j];
            }
        }
    }

    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < w; j++)
        {
            iTemp = 0;
            for(int k = 1; k < c; k++)
            {
                if(ncnnMatOUT[k*h*w + i*w + j] > ncnnMatOUT[iTemp*h*w + i*w + j])
                {
                    iTemp = k;
                }
                else
                {
                    continue;
                }
            }
            if(iTemp == (c-1))
            {
                loc[i*w + j] = 0;
                stLaneRslt.afLaneRslt[i][j] = 0;
            }
            else
            {

                //stLaneRslt.afLaneRslt[i][j] = ((float)iTemp + 1) * flColSampleW;
                stLaneRslt.afLaneRslt[i][j] = loc[i*w + j] * flColSampleW;
            }
        }
    }



    endTime = clock();
    std::cout<<"time is: "<<(double)(endTime - startTime) / CLOCKS_PER_SEC<<"s"<<std::endl;

    // for(int i = 0; i < h; i++)
    // {
    //     for(int j = 0; j < w; j++)
    //     {
    //         std::cout<<stLaneRslt.afLaneRslt[i][j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    delete [] loc;
    delete [] prob;
    delete [] prob_sum;
    ex.clear();
    ncnnMatOUT.release();
}
*/

int main()
{
    MODEL_CONFIG stModelConfig, *pstModelConfig = &stModelConfig;
    stModelConfig.iInputImgH = 288;
    stModelConfig.iInputImgW = 800;
    stModelConfig.afMean[0] = 0.485*255;
    stModelConfig.afMean[1] = 0.456*255;
    stModelConfig.afMean[2] = 0.406*255;
    stModelConfig.afNorm[0] = 1/0.229/255;
    stModelConfig.afNorm[1] = 1/0.224/255;
    stModelConfig.afNorm[2] = 1/0.225/255;
    stModelConfig.flColSampleW = 799 / 63;
    stModelConfig.aiRowAnchor[0] = 121;
    stModelConfig.aiRowAnchor[1] = 131;
    stModelConfig.aiRowAnchor[2] = 141;
    stModelConfig.aiRowAnchor[3] = 150;
    stModelConfig.aiRowAnchor[4] = 160;
    stModelConfig.aiRowAnchor[5] = 170;
    stModelConfig.aiRowAnchor[6] = 180;
    stModelConfig.aiRowAnchor[7] = 189;
    stModelConfig.aiRowAnchor[8] = 199;
    stModelConfig.aiRowAnchor[9] = 209;
    stModelConfig.aiRowAnchor[10] = 219;
    stModelConfig.aiRowAnchor[11] = 228;
    stModelConfig.aiRowAnchor[12] = 238;
    stModelConfig.aiRowAnchor[13] = 248;
    stModelConfig.aiRowAnchor[14] = 258;
    stModelConfig.aiRowAnchor[15] = 267;
    stModelConfig.aiRowAnchor[16] = 277;
    stModelConfig.aiRowAnchor[17] = 287;
    stModelConfig.pacModelbinPath = "../model/UFAST_culane_50.bin";
    stModelConfig.pacModelParamPath = "../model/UFAST_culane_50.param";

    cv::Mat img = cv::imread("../img/0.jpg");

    clModel clMyModel(pstModelConfig);
    //多执行几次看看是否正常
    for(int i = 0; i < 10; i++)
    {
        clMyModel.GetModelRslt(img);
    }
    cv::Point2i LanePoint;

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 18; j++)
        {
            if(clMyModel.stLaneRslt.afLaneRslt[j][i] > 0)
            {
                LanePoint.x = int(clMyModel.stLaneRslt.afLaneRslt[j][i] * 640 / 800);
                LanePoint.y = int(stModelConfig.aiRowAnchor[j] * 360 / 288);
                cv::circle(img, LanePoint, 5, (0,255,0), 5);
            }
        }
    }

    cv::imshow("Lane", img);
    cv::waitKey(0);

    return 0;
}
