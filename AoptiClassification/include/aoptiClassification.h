#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//导出类为dll
#ifndef DLL_API
#define DLL_API __declspec(dllexport)
#endif

class DLL_API AoptiClassification
{
private:
	//模型dll文件路径
	std::string lib_path;

	//网络结构json文件路径
	std::string json_path;

	//网络参数文件路径
	std::string param_path;

	//分类数量
	int num_classes;

	//错误信息
	std::string error_message;

	//模型函数和读取参数
	tvm::runtime::PackedFunc set_input;
	tvm::runtime::PackedFunc run;
	tvm::runtime::PackedFunc get_output;

	//输入和输出
	DLTensor* x;
	DLTensor* y;

	//图片预处理
	virtual void ProcessData(cv::Mat &img);

public:
	AoptiClassification(void);
	
	//初始化模型
	bool InitClassification(std::string pretrainedModelPath, std::string pretrainedModelName, int num_classes);

	//获取错误信息
	const std::string GetErrorMessage();

	//获取分类结果
	int Classification(uchar *pImg, int nHeight, int nWidth);

	//析构函数，释放内存
	~AoptiClassification();
};

